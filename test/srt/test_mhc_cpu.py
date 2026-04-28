"""Tests for CPU MHC (Multi-Head Channel) kernels.

Verifies that the pure-PyTorch CPU implementations in mhc_cpu.py
produce numerically equivalent results to:
  - The reference torch fallback paths in deepseek_v4.py
  - Mathematical invariants (doubly-stochastic comb matrix)

Run:
    python test/srt/test_mhc_cpu.py
    pytest  test/srt/test_mhc_cpu.py -v
"""

import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.srt.layers.mhc_cpu import (
    hc_head_cpu,
    hc_post_cpu,
    hc_pre_cpu,
    hc_split_sinkhorn_cpu,
)

# Powers-of-two T values from decode (1) up to large prefill (2048).
T_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# ---------------------------------------------------------------------------
# Reference implementations (identical to the torch fallbacks in
# DeepseekV4DecoderLayer and DeepseekV4Model so tests are self-contained)
# ---------------------------------------------------------------------------


def _ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Reference: naive per-token loop matching the TileLang kernel exactly."""
    T, mix_hc = mixes.shape
    hc = hc_mult

    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )

    comb = mixes[:, 2 * hc :].view(T, hc, hc) * hc_scale[2] + hc_base[2 * hc :].view(
        hc, hc
    )

    # Row-stable exp
    comb = (comb - comb.amax(dim=-1, keepdim=True)).exp()

    # First Sinkhorn iteration (mirrors TileLang kernel)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)

    return pre, post, comb


def _ref_hc_pre(
    x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_norm_eps, hc_eps
):
    """Reference: torch fallback from DeepseekV4DecoderLayer.hc_pre."""
    dtype = x.dtype
    shape = x.size()
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt  # [T, mix_hc]

    pre, post, comb = _ref_hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
    )
    y = (pre.unsqueeze(-1) * x.float()).sum(dim=1)
    return y.to(dtype), post, comb


def _ref_hc_post(x, residual, post, comb):
    """Reference: torch fallback from DeepseekV4DecoderLayer.hc_post."""
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


def _ref_hc_head(x, hc_fn, hc_scale, hc_base, hc_eps, norm_eps):
    """Reference: torch implementation from DeepseekV4Model.hc_head."""
    dtype = x.dtype
    shape = x.size()
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape).float(), dim=1)
    return y.to(dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(T, hc, d, dtype=torch.float32, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    mix_hc = (2 + hc) * hc

    x = torch.randn(T, hc, d, dtype=dtype, generator=gen)
    hc_fn = torch.randn(mix_hc, hc * d, dtype=dtype, generator=gen) * 0.02
    hc_scale = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
    hc_base = torch.zeros(mix_hc, dtype=torch.float32)
    return x, hc_fn, hc_scale, hc_base


def _make_post_inputs(T, hc, d, dtype=torch.float32, seed=7):
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.randn(T, d, dtype=dtype, generator=gen)
    residual = torch.randn(T, hc, d, dtype=dtype, generator=gen)
    post = torch.rand(T, hc, dtype=torch.float32, generator=gen)
    comb = torch.rand(T, hc, hc, dtype=torch.float32, generator=gen)
    # Normalise comb to be roughly doubly stochastic
    comb = comb / comb.sum(dim=-1, keepdim=True)
    comb = comb / comb.sum(dim=-2, keepdim=True)
    return x, residual, post, comb


# ---------------------------------------------------------------------------
# Tests: hc_split_sinkhorn_cpu
# ---------------------------------------------------------------------------


class TestHcSplitSinkhornCpu(unittest.TestCase):
    """Verify hc_split_sinkhorn_cpu against the reference."""

    def _run(self, T, hc, dtype, sinkhorn_iters, eps, tol):
        mix_hc = (2 + hc) * hc
        gen = torch.Generator()
        gen.manual_seed(0)
        mixes = torch.randn(T, mix_hc, dtype=torch.float32, generator=gen)
        hc_scale = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
        hc_base = torch.zeros(mix_hc, dtype=torch.float32)

        pre_ref, post_ref, comb_ref = _ref_hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps
        )
        pre_cpu, post_cpu, comb_cpu = hc_split_sinkhorn_cpu(
            mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps
        )

        torch.testing.assert_close(pre_cpu, pre_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(post_cpu, post_ref, atol=tol, rtol=tol)
        torch.testing.assert_close(comb_cpu, comb_ref, atol=tol, rtol=tol)

    def test_decode_t1(self):
        self._run(T=1, hc=4, dtype=torch.float32, sinkhorn_iters=20, eps=1e-6, tol=1e-5)

    def test_prefill_t32(self):
        self._run(
            T=32, hc=4, dtype=torch.float32, sinkhorn_iters=20, eps=1e-6, tol=1e-5
        )

    def test_prefill_t256(self):
        self._run(
            T=256, hc=4, dtype=torch.float32, sinkhorn_iters=20, eps=1e-6, tol=1e-5
        )

    def test_sinkhorn_iters_1(self):
        self._run(T=16, hc=4, dtype=torch.float32, sinkhorn_iters=1, eps=1e-6, tol=1e-5)

    def test_hc2(self):
        self._run(
            T=16, hc=2, dtype=torch.float32, sinkhorn_iters=20, eps=1e-6, tol=1e-5
        )

    def test_output_shapes(self):
        T, hc, mix_hc = 8, 4, (2 + 4) * 4
        mixes = torch.randn(T, mix_hc)
        hc_scale = torch.ones(3)
        hc_base = torch.zeros(mix_hc)
        pre, post, comb = hc_split_sinkhorn_cpu(mixes, hc_scale, hc_base, hc)
        self.assertEqual(pre.shape, (T, hc))
        self.assertEqual(post.shape, (T, hc))
        self.assertEqual(comb.shape, (T, hc, hc))

    def test_pre_range(self):
        """pre values must be in (eps, 1+eps] since sigmoid ∈ (0,1)."""
        T, hc, mix_hc, eps = 64, 4, 24, 1e-6
        mixes = torch.randn(T, mix_hc) * 5
        hc_scale = torch.ones(3)
        hc_base = torch.zeros(mix_hc)
        pre, _, _ = hc_split_sinkhorn_cpu(mixes, hc_scale, hc_base, hc, eps=eps)
        self.assertTrue((pre >= eps).all(), "pre must be >= eps")
        self.assertTrue((pre <= 1 + eps).all(), "pre must be <= 1+eps")

    def test_post_range(self):
        """post values must be in [0, 2] numerically (2*sigmoid may saturate to 2)."""
        T, hc, mix_hc = 64, 4, 24
        mixes = torch.randn(T, mix_hc) * 5
        hc_scale = torch.ones(3)
        hc_base = torch.zeros(mix_hc)
        _, post, _ = hc_split_sinkhorn_cpu(mixes, hc_scale, hc_base, hc)
        self.assertTrue((post >= 0).all(), "post must be non-negative")
        self.assertTrue((post <= 2).all(), "post must be <= 2")

    def test_comb_approx_doubly_stochastic(self):
        """After sinkhorn, comb row-sums and col-sums should be approximately equal."""
        T, hc, mix_hc = 32, 4, 24
        mixes = torch.randn(T, mix_hc)
        hc_scale = torch.ones(3)
        hc_base = torch.zeros(mix_hc)
        _, _, comb = hc_split_sinkhorn_cpu(mixes, hc_scale, hc_base, hc)
        row_sums = comb.sum(dim=-1)  # [T, hc]
        col_sums = comb.sum(dim=-2)  # [T, hc]
        # They won't be exactly 1 due to eps, but all row-sums should be close
        # to each other and all col-sums should be close to each other
        row_std = row_sums.std(dim=-1)  # std across hc for each token
        col_std = col_sums.std(dim=-1)
        self.assertTrue((row_std < 1e-2).all(), "row sums should be nearly uniform")
        self.assertTrue((col_std < 1e-2).all(), "col sums should be nearly uniform")

    def test_t_sweep(self):
        """Cover T=1..2048 (powers of two) to catch boundary / tiling bugs."""
        hc, mix_hc = 4, 24
        hc_scale = torch.ones(3)
        hc_base = torch.zeros(mix_hc)
        for T in T_SWEEP:
            with self.subTest(T=T):
                gen = torch.Generator()
                gen.manual_seed(T)
                mixes = torch.randn(T, mix_hc, generator=gen)
                pre_ref, post_ref, comb_ref = _ref_hc_split_sinkhorn(
                    mixes, hc_scale, hc_base, hc, 20, 1e-6
                )
                pre_cpu, post_cpu, comb_cpu = hc_split_sinkhorn_cpu(
                    mixes, hc_scale, hc_base, hc, 20, 1e-6
                )
                torch.testing.assert_close(pre_cpu, pre_ref, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(post_cpu, post_ref, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(comb_cpu, comb_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: hc_pre_cpu
# ---------------------------------------------------------------------------


class TestHcPreCpu(unittest.TestCase):
    """Verify hc_pre_cpu against the torch reference."""

    HC = 4
    D = 128
    REAL_HIDDEN_SIZES = (4096, 7168)
    RMS_EPS = 1e-5
    HC_EPS = 1e-6
    SINKHORN_ITERS = 20

    def _run(self, T, dtype, atol):
        x, hc_fn, hc_scale, hc_base = _make_inputs(T, self.HC, self.D, dtype=dtype)

        y_ref, post_ref, comb_ref = _ref_hc_pre(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.HC,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )
        y_cpu, post_cpu, comb_cpu = hc_pre_cpu(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.HC,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )

        torch.testing.assert_close(y_cpu, y_ref, atol=atol, rtol=atol)
        torch.testing.assert_close(post_cpu, post_ref, atol=atol, rtol=atol)
        torch.testing.assert_close(comb_cpu, comb_ref, atol=atol, rtol=atol)

    def test_decode_fp32(self):
        self._run(T=1, dtype=torch.float32, atol=1e-5)

    def test_prefill_fp32(self):
        self._run(T=64, dtype=torch.float32, atol=1e-5)

    def test_decode_bf16(self):
        # bf16 loses precision; allow looser tolerance
        self._run(T=1, dtype=torch.bfloat16, atol=5e-3)

    def test_prefill_bf16(self):
        self._run(T=64, dtype=torch.bfloat16, atol=5e-3)

    def test_output_shapes(self):
        T, hc, d = 8, self.HC, self.D
        x, hc_fn, hc_scale, hc_base = _make_inputs(T, hc, d)
        y, post, comb = hc_pre_cpu(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )
        self.assertEqual(y.shape, (T, d))
        self.assertEqual(post.shape, (T, hc))
        self.assertEqual(comb.shape, (T, hc, hc))

    def test_output_dtype_preserved_fp32(self):
        x, hc_fn, hc_scale, hc_base = _make_inputs(
            8, self.HC, self.D, dtype=torch.float32
        )
        y, _, _ = hc_pre_cpu(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.HC,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )
        self.assertEqual(y.dtype, torch.float32)

    def test_output_dtype_preserved_bf16(self):
        x, hc_fn, hc_scale, hc_base = _make_inputs(
            8, self.HC, self.D, dtype=torch.bfloat16
        )
        y, _, _ = hc_pre_cpu(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.HC,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_empty_tokens(self):
        """T=0 should not crash and return empty tensors."""
        hc, d = self.HC, self.D
        mix_hc = (2 + hc) * hc
        x = torch.empty(0, hc, d, dtype=torch.float32)
        hc_fn = torch.randn(mix_hc, hc * d)
        hc_scale = torch.ones(3)
        hc_base = torch.zeros(mix_hc)
        y, post, comb = hc_pre_cpu(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )
        self.assertEqual(y.shape[0], 0)
        self.assertEqual(post.shape[0], 0)
        self.assertEqual(comb.shape[0], 0)

    def test_large_prefill(self):
        """Smoke test with T=512 to catch shape/indexing issues."""
        self._run(T=512, dtype=torch.float32, atol=1e-5)

    def test_real_shape_decode_fp32(self):
        """DeepSeekV4-real hidden sizes in decode shape with fp32."""
        for d in self.REAL_HIDDEN_SIZES:
            x, hc_fn, hc_scale, hc_base = _make_inputs(
                1, self.HC, d, dtype=torch.float32
            )
            y_ref, post_ref, comb_ref = _ref_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )
            y_cpu, post_cpu, comb_cpu = hc_pre_cpu(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )
            torch.testing.assert_close(y_cpu, y_ref, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(post_cpu, post_ref, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(comb_cpu, comb_ref, atol=1e-5, rtol=1e-5)

    def test_real_shape_prefill_bf16(self):
        """DeepSeekV4-real hidden sizes in prefill shape with bf16."""
        for d in self.REAL_HIDDEN_SIZES:
            x, hc_fn, hc_scale, hc_base = _make_inputs(
                8, self.HC, d, dtype=torch.bfloat16
            )
            y_ref, post_ref, comb_ref = _ref_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )
            y_cpu, post_cpu, comb_cpu = hc_pre_cpu(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )
            torch.testing.assert_close(y_cpu, y_ref, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(post_cpu, post_ref, atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(comb_cpu, comb_ref, atol=5e-3, rtol=5e-3)

    def test_t_sweep_fp32(self):
        """Cover T=1..2048 (powers of two) with fp32."""
        for T in T_SWEEP:
            with self.subTest(T=T):
                self._run(T, dtype=torch.float32, atol=1e-5)

    def test_t_sweep_bf16(self):
        """Cover T=1..2048 (powers of two) with bf16."""
        for T in T_SWEEP:
            with self.subTest(T=T):
                self._run(T, dtype=torch.bfloat16, atol=5e-3)


# ---------------------------------------------------------------------------
# Tests: hc_post_cpu
# ---------------------------------------------------------------------------


class TestHcPostCpu(unittest.TestCase):
    """Verify hc_post_cpu against the torch reference."""

    HC = 4
    D = 128
    REAL_HIDDEN_SIZES = (4096, 7168)

    def _run(self, T, dtype, atol):
        x, residual, post, comb = _make_post_inputs(T, self.HC, self.D, dtype=dtype)
        out_ref = _ref_hc_post(x, residual, post, comb)
        out_cpu = hc_post_cpu(x, residual, post, comb)
        torch.testing.assert_close(out_cpu, out_ref, atol=atol, rtol=atol)

    def test_decode_fp32(self):
        self._run(T=1, dtype=torch.float32, atol=1e-5)

    def test_prefill_fp32(self):
        self._run(T=64, dtype=torch.float32, atol=1e-5)

    def test_decode_bf16(self):
        self._run(T=1, dtype=torch.bfloat16, atol=5e-3)

    def test_prefill_bf16(self):
        self._run(T=64, dtype=torch.bfloat16, atol=5e-3)

    def test_output_shape(self):
        T, hc, d = 8, self.HC, self.D
        x, residual, post, comb = _make_post_inputs(T, hc, d)
        out = hc_post_cpu(x, residual, post, comb)
        self.assertEqual(out.shape, (T, hc, d))

    def test_output_dtype_preserved_fp32(self):
        x, residual, post, comb = _make_post_inputs(
            8, self.HC, self.D, dtype=torch.float32
        )
        out = hc_post_cpu(x, residual, post, comb)
        self.assertEqual(out.dtype, torch.float32)

    def test_output_dtype_preserved_bf16(self):
        x, residual, post, comb = _make_post_inputs(
            8, self.HC, self.D, dtype=torch.bfloat16
        )
        out = hc_post_cpu(x, residual, post, comb)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_zero_post_means_only_residual(self):
        """If post is zero, output equals comb^T @ residual (no sublayer contribution)."""
        T, hc, d = 4, self.HC, self.D
        gen = torch.Generator()
        gen.manual_seed(99)
        x = torch.randn(T, d, generator=gen)
        residual = torch.randn(T, hc, d, generator=gen)
        post = torch.zeros(T, hc)
        comb = torch.eye(hc).unsqueeze(0).expand(T, hc, hc)  # identity comb

        out = hc_post_cpu(x, residual, post, comb)
        # With identity comb and zero post: out[t, j, k] = sum_i δ(i,j) * residual[t,i,k]
        # => out[t, j, k] = residual[t, j, k]
        torch.testing.assert_close(out.float(), residual.float(), atol=1e-6, rtol=1e-6)

    def test_identity_comb_unit_post(self):
        """Identity comb + unit post: output = x (broadcast) + residual."""
        T, hc, d = 4, self.HC, self.D
        gen = torch.Generator()
        gen.manual_seed(13)
        x = torch.randn(T, d, generator=gen)
        residual = torch.randn(T, hc, d, generator=gen)
        post = torch.ones(T, hc)
        comb = torch.eye(hc).unsqueeze(0).expand(T, hc, hc)

        out = hc_post_cpu(x, residual, post, comb)
        expected = x.unsqueeze(1) + residual  # [T, hc, d]
        torch.testing.assert_close(out.float(), expected.float(), atol=1e-5, rtol=1e-5)

    def test_large_prefill(self):
        self._run(T=512, dtype=torch.float32, atol=1e-5)

    def test_real_shape_decode_fp32(self):
        """DeepSeekV4-real hidden sizes in decode shape with fp32."""
        for d in self.REAL_HIDDEN_SIZES:
            x, residual, post, comb = _make_post_inputs(
                1, self.HC, d, dtype=torch.float32
            )
            out_ref = _ref_hc_post(x, residual, post, comb)
            out_cpu = hc_post_cpu(x, residual, post, comb)
            torch.testing.assert_close(out_cpu, out_ref, atol=1e-5, rtol=1e-5)

    def test_real_shape_prefill_bf16(self):
        """DeepSeekV4-real hidden sizes in prefill shape with bf16."""
        for d in self.REAL_HIDDEN_SIZES:
            x, residual, post, comb = _make_post_inputs(
                8, self.HC, d, dtype=torch.bfloat16
            )
            out_ref = _ref_hc_post(x, residual, post, comb)
            out_cpu = hc_post_cpu(x, residual, post, comb)
            torch.testing.assert_close(out_cpu, out_ref, atol=5e-3, rtol=5e-3)

    def test_t_sweep_fp32(self):
        """Cover T=1..2048 (powers of two) with fp32."""
        for T in T_SWEEP:
            with self.subTest(T=T):
                self._run(T, dtype=torch.float32, atol=1e-5)

    def test_t_sweep_bf16(self):
        """Cover T=1..2048 (powers of two) with bf16."""
        for T in T_SWEEP:
            with self.subTest(T=T):
                self._run(T, dtype=torch.bfloat16, atol=5e-3)


# ---------------------------------------------------------------------------
# Tests: hc_head_cpu
# ---------------------------------------------------------------------------


class TestHcHeadCpu(unittest.TestCase):
    """Verify hc_head_cpu against the torch reference."""

    HC = 4
    D = 128
    REAL_HIDDEN_SIZES = (4096, 7168)
    HC_EPS = 1e-6
    NORM_EPS = 1e-5

    def _make(self, T, d=None, dtype=torch.float32, seed=0):
        gen = torch.Generator()
        gen.manual_seed(seed)
        d = self.D if d is None else d
        x = torch.randn(T, self.HC, d, dtype=dtype, generator=gen)
        hc_fn = torch.randn(self.HC, self.HC * d, dtype=dtype, generator=gen) * 0.02
        hc_scale = torch.tensor(1.0, dtype=torch.float32)
        hc_base = torch.zeros(self.HC, dtype=torch.float32)
        return x, hc_fn, hc_scale, hc_base

    def _run(self, T, dtype, atol):
        x, hc_fn, hc_scale, hc_base = self._make(T, dtype=dtype)
        ref = _ref_hc_head(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
        cpu = hc_head_cpu(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
        torch.testing.assert_close(cpu, ref, atol=atol, rtol=atol)

    def test_decode_fp32(self):
        self._run(T=1, dtype=torch.float32, atol=1e-5)

    def test_prefill_fp32(self):
        self._run(T=64, dtype=torch.float32, atol=1e-5)

    def test_decode_bf16(self):
        self._run(T=1, dtype=torch.bfloat16, atol=5e-3)

    def test_prefill_bf16(self):
        self._run(T=64, dtype=torch.bfloat16, atol=5e-3)

    def test_output_shape(self):
        x, hc_fn, hc_scale, hc_base = self._make(8)
        y = hc_head_cpu(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
        self.assertEqual(y.shape, (8, self.D))

    def test_output_dtype_preserved_fp32(self):
        x, hc_fn, hc_scale, hc_base = self._make(4, dtype=torch.float32)
        y = hc_head_cpu(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
        self.assertEqual(y.dtype, torch.float32)

    def test_output_dtype_preserved_bf16(self):
        x, hc_fn, hc_scale, hc_base = self._make(4, dtype=torch.bfloat16)
        y = hc_head_cpu(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_real_shape_decode_fp32(self):
        """DeepSeekV4-real hidden sizes in decode shape with fp32."""
        for d in self.REAL_HIDDEN_SIZES:
            x, hc_fn, hc_scale, hc_base = self._make(1, d, dtype=torch.float32)
            ref = _ref_hc_head(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
            cpu = hc_head_cpu(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
            torch.testing.assert_close(cpu, ref, atol=1e-5, rtol=1e-5)

    def test_real_shape_prefill_bf16(self):
        """DeepSeekV4-real hidden sizes in prefill shape with bf16."""
        for d in self.REAL_HIDDEN_SIZES:
            x, hc_fn, hc_scale, hc_base = self._make(8, d, dtype=torch.bfloat16)
            ref = _ref_hc_head(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
            cpu = hc_head_cpu(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)
            # Large real shapes in bf16 can differ by one bf16 quantization step.
            torch.testing.assert_close(cpu, ref, atol=2e-2, rtol=1e-2)

    def test_t_sweep_fp32(self):
        """Cover T=1..2048 (powers of two) with fp32."""
        for T in T_SWEEP:
            with self.subTest(T=T):
                self._run(T, dtype=torch.float32, atol=1e-5)

    def test_t_sweep_bf16(self):
        """Cover T=1..2048 (powers of two) with bf16."""
        for T in T_SWEEP:
            with self.subTest(T=T):
                self._run(T, dtype=torch.bfloat16, atol=5e-3)


# ---------------------------------------------------------------------------
# Integration: full pre→post round-trip
# ---------------------------------------------------------------------------


class TestMhcRoundTrip(unittest.TestCase):
    """End-to-end pre→post round-trip shapes and dtype consistency."""

    HC = 4
    D = 128
    REAL_HIDDEN_SIZES = (4096, 7168)
    RMS_EPS = 1e-5
    HC_EPS = 1e-6
    SINKHORN_ITERS = 20

    def _round_trip(self, T, dtype):
        x, hc_fn, hc_scale, hc_base = _make_inputs(T, self.HC, self.D, dtype=dtype)

        # hc_pre
        y, post, comb = hc_pre_cpu(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.HC,
            self.SINKHORN_ITERS,
            self.RMS_EPS,
            self.HC_EPS,
        )
        self.assertEqual(y.shape, (T, self.D))
        self.assertEqual(post.shape, (T, self.HC))
        self.assertEqual(comb.shape, (T, self.HC, self.HC))
        self.assertEqual(y.dtype, dtype)

        # Simulate a sublayer: identity (x = y)
        sublayer_out = y  # [T, d]

        # hc_post
        out = hc_post_cpu(sublayer_out, x, post, comb)
        self.assertEqual(out.shape, (T, self.HC, self.D))
        self.assertEqual(out.dtype, dtype)

    def test_decode_fp32(self):
        self._round_trip(T=1, dtype=torch.float32)

    def test_prefill_fp32(self):
        self._round_trip(T=64, dtype=torch.float32)

    def test_decode_bf16(self):
        self._round_trip(T=1, dtype=torch.bfloat16)

    def test_prefill_bf16(self):
        self._round_trip(T=64, dtype=torch.bfloat16)

    def test_real_shape_decode_fp32(self):
        """Round-trip correctness with DeepSeekV4-real hidden sizes."""
        for d in self.REAL_HIDDEN_SIZES:
            x, hc_fn, hc_scale, hc_base = _make_inputs(
                1, self.HC, d, dtype=torch.float32
            )
            y, post, comb = hc_pre_cpu(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )
            out = hc_post_cpu(y, x, post, comb)

            y_ref, post_ref, comb_ref = _ref_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )
            out_ref = _ref_hc_post(y_ref, x, post_ref, comb_ref)

            self.assertEqual(y.shape, (1, d))
            self.assertEqual(out.shape, (1, self.HC, d))
            torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Against-CUDA golden tests
# (skipped when CUDA not available; provide the strongest correctness guarantee)
#
# Why these tests matter
# ----------------------
# _ref_hc_split_sinkhorn is a manual Python transcription of the TileLang
# kernel.  The tests above (TestHcSplitSinkhornCpu, TestHcPreCpu) only verify
# that hc_split_sinkhorn_cpu matches _ref_hc_split_sinkhorn – a circular check
# if both are derived from the same source.
#
# TestAgainstCudaKernel breaks the circularity by comparing against the actual
# TileLang GPU kernel.  When these tests pass we know:
#   _ref_hc_split_sinkhorn ≈ CUDA kernel  (test_ref_hc_split_sinkhorn_vs_cuda)
#   hc_split_sinkhorn_cpu  ≈ CUDA kernel  (test_hc_split_sinkhorn_cpu_vs_cuda)
# which transitively validates every test that relies on _ref_hc_split_sinkhorn.
#
# Shape contract
# --------------
# The CUDA path in deepseek_v4.py calls hc_split_sinkhorn with mixes shaped
# [T, 1, mix_hc] (b=T, s=1), getting back [T, 1, hc] / [T, 1, hc, hc], then
# squeezes the s-dimension.  The CPU path uses [T, mix_hc] directly and returns
# [T, hc] / [T, hc, hc] – the same shape after the CUDA squeeze.
# test_shapes_from_cuda_path makes this contract explicit.
# ---------------------------------------------------------------------------

_CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA not available – skipping golden tests")
class TestAgainstCudaKernel(unittest.TestCase):
    """Compare CPU kernels against TileLang GPU kernels (authoritative golden ref)."""

    HC = 4
    SINKHORN_ITERS = 20
    EPS = 1e-6

    def _sinkhorn_inputs(self, T, seed=42):
        hc, mix_hc = self.HC, (2 + self.HC) * self.HC
        gen = torch.Generator()
        gen.manual_seed(seed)
        mixes = torch.randn(T, mix_hc, generator=gen)
        hc_scale = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
        hc_base = torch.zeros(mix_hc, dtype=torch.float32)
        return mixes, hc_scale, hc_base

    def _cuda_sinkhorn(self, mixes, hc_scale, hc_base):
        """Run TileLang GPU kernel on [T, mix_hc] input; return squeezed [T, hc] output."""
        from sglang.srt.layers.mhc import hc_split_sinkhorn as cuda_sinkhorn

        # hc_split_sinkhorn expects [b, s, mix_hc]; wrap with s=1 then squeeze
        pre, post, comb = cuda_sinkhorn(
            mixes.cuda().unsqueeze(1),
            hc_scale.cuda(),
            hc_base.cuda(),
            self.HC,
            self.SINKHORN_ITERS,
            self.EPS,
        )
        return pre.squeeze(1).cpu(), post.squeeze(1).cpu(), comb.squeeze(1).cpu()

    def test_hc_split_sinkhorn_cpu_vs_cuda(self):
        """hc_split_sinkhorn_cpu must match TileLang GPU kernel output."""
        mixes, hc_scale, hc_base = self._sinkhorn_inputs(T=64)
        pre_cpu, post_cpu, comb_cpu = hc_split_sinkhorn_cpu(
            mixes,
            hc_scale,
            hc_base,
            self.HC,
            self.SINKHORN_ITERS,
            self.EPS,
        )
        pre_gpu, post_gpu, comb_gpu = self._cuda_sinkhorn(mixes, hc_scale, hc_base)

        tol = 1e-5
        torch.testing.assert_close(pre_cpu, pre_gpu, atol=tol, rtol=tol)
        torch.testing.assert_close(post_cpu, post_gpu, atol=tol, rtol=tol)
        torch.testing.assert_close(comb_cpu, comb_gpu, atol=tol, rtol=tol)

    def test_ref_hc_split_sinkhorn_vs_cuda(self):
        """Validate _ref_hc_split_sinkhorn against the CUDA kernel.

        This closes the proof chain: once _ref_hc_split_sinkhorn ≈ CUDA kernel,
        all tests that use it as a reference are no longer circular.
        """
        mixes, hc_scale, hc_base = self._sinkhorn_inputs(T=64, seed=7)
        pre_ref, post_ref, comb_ref = _ref_hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.HC,
            self.SINKHORN_ITERS,
            self.EPS,
        )
        pre_gpu, post_gpu, comb_gpu = self._cuda_sinkhorn(mixes, hc_scale, hc_base)

        tol = 1e-5
        torch.testing.assert_close(pre_ref, pre_gpu, atol=tol, rtol=tol)
        torch.testing.assert_close(post_ref, post_gpu, atol=tol, rtol=tol)
        torch.testing.assert_close(comb_ref, comb_gpu, atol=tol, rtol=tol)

    def test_shapes_from_cuda_path(self):
        """GPU kernel output shapes, after squeezing s=1, must equal CPU contract.

        CUDA path: mixes [T,1,mix_hc] → pre/post [T,1,hc], comb [T,1,hc,hc]
        CPU path:  mixes [T,mix_hc]   → pre/post [T,hc],   comb [T,hc,hc]
        After .squeeze(1) both are identical.
        """
        from sglang.srt.layers.mhc import hc_split_sinkhorn as cuda_sinkhorn

        T, hc = 32, self.HC
        mixes, hc_scale, hc_base = self._sinkhorn_inputs(T)
        pre_g, post_g, comb_g = cuda_sinkhorn(
            mixes.cuda().unsqueeze(1),
            hc_scale.cuda(),
            hc_base.cuda(),
            hc,
            self.SINKHORN_ITERS,
            self.EPS,
        )
        # Raw GPU shapes include s=1 dim
        self.assertEqual(pre_g.shape, (T, 1, hc))
        self.assertEqual(post_g.shape, (T, 1, hc))
        self.assertEqual(comb_g.shape, (T, 1, hc, hc))
        # After squeeze: matches CPU contract
        self.assertEqual(pre_g.squeeze(1).shape, (T, hc))
        self.assertEqual(post_g.squeeze(1).shape, (T, hc))
        self.assertEqual(comb_g.squeeze(1).shape, (T, hc, hc))


# ---------------------------------------------------------------------------
# Performance Benchmarks
# (These tests measure kernel performance to guide optimization efforts)

# Usage:
#   python test/srt/test_mhc_cpu.py BenchmarkHcPreCpu   (run hc_pre benchmarks)
#   python test/srt/test_mhc_cpu.py BenchmarkHcPostCpu  (run hc_post benchmarks)
#   python test/srt/test_mhc_cpu.py BenchmarkHcHeadCpu  (run hc_head benchmarks)
#   python test/srt/test_mhc_cpu.py Benchmark           (run all benchmarks)

# Benchmark structure:
#   1. Vary sequence length T to measure decode vs prefill scalability
#   2. Measure C++ kernel vs PyTorch fallback to quantify optimization gains
#   3. Report wall-clock time, throughput (elements/ms), and speedup ratio
# ---------------------------------------------------------------------------

import time


class BenchmarkHelper:
    """Utilities for consistent performance measurement."""

    @staticmethod
    def timeit(func, warmup_iters=3, measure_iters=5, description=""):
        """
        Run func with warmup, then measure wall-clock time.
        Returns (mean_time_ms, std_time_ms, times_list).
        """
        # Warmup
        for _ in range(warmup_iters):
            func()

        # Measurement
        times = []
        for _ in range(measure_iters):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            func()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5

        return mean_ms, std_ms, times

    @staticmethod
    def report_benchmark(name, cpp_time_ms, py_time_ms, throughput_cpp, throughput_py):
        """Format and print a single benchmark result."""
        speedup = py_time_ms / cpp_time_ms if cpp_time_ms > 0 else 1.0
        print(
            f"  {name:40s} | "
            f"C++ {cpp_time_ms:7.3f}ms | "
            f"PyTorch {py_time_ms:7.3f}ms | "
            f"Speedup {speedup:5.2f}x | "
            f"Throughput (C++/PyTorch): {throughput_cpp:7.1f}/{throughput_py:7.1f} elem/ms"
        )


def _bench_iters(T, d):
    """Return (warmup, measure) iteration counts scaled to keep each point ~1-2s.

    Large T×d means each call is expensive; reduce iter counts to stay fast.
    Small T×d (decode) needs many iters for stable timing.
    """
    cost = T * max(d, 1)
    if cost <= 4096:  # T=1, small d
        return 500, 500
    elif cost <= 32 * 4096:  # T<=32
        return 200, 200
    elif cost <= 128 * 4096:  # T<=128
        return 100, 100
    elif cost <= 512 * 4096:  # T<=512
        return 30, 50
    else:  # T=1024, 2048
        return 10, 20


class BenchmarkHcSplitSinkhornCpu(unittest.TestCase):
    """Benchmark hc_split_sinkhorn_cpu vs reference implementation."""

    HC = 4

    def _run_benchmark(self, T, hc, description, warmup=50, measure=100):
        """Benchmark hc_split_sinkhorn for a given T."""
        mix_hc = (2 + hc) * hc
        mixes = torch.randn(T, mix_hc)
        hc_scale = torch.ones(3)
        hc_base = torch.zeros(mix_hc)

        # C++ kernel
        def run_cpp():
            hc_split_sinkhorn_cpu(mixes, hc_scale, hc_base, hc, sinkhorn_iters=20)

        # PyTorch reference
        def run_py():
            _ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 20, eps=1e-6)

        cpp_time, cpp_std, _ = BenchmarkHelper.timeit(
            run_cpp, warmup_iters=warmup, measure_iters=measure
        )
        py_time, py_std, _ = BenchmarkHelper.timeit(
            run_py, warmup_iters=warmup, measure_iters=measure
        )

        # Throughput: elements processed
        # mixes: T x mix_hc; output: pre (T x hc) + post (T x hc) + comb (T x hc x hc)
        total_elements = T * mix_hc + T * hc + T * hc + T * hc * hc
        throughput_cpp = total_elements / cpp_time
        throughput_py = total_elements / py_time

        BenchmarkHelper.report_benchmark(
            description, cpp_time, py_time, throughput_cpp, throughput_py
        )

    def test_benchmark_sweep(self):
        """Sweep T=1..2048 (powers of two)."""
        print()
        for T in T_SWEEP:
            w, m = _bench_iters(T, d=1)
            self._run_benchmark(
                T,
                hc=4,
                description=f"hc_split_sinkhorn (T={T:4d})",
                warmup=w,
                measure=m,
            )


class BenchmarkHcPreCpu(unittest.TestCase):
    """Benchmark hc_pre_cpu vs PyTorch reference."""

    HC = 4
    HIDDEN_SIZES = (4096, 7168)
    RMS_EPS = 1e-5
    HC_EPS = 1e-6
    SINKHORN_ITERS = 20

    def _run_benchmark(self, T, d, dtype, description, warmup=500, measure=500):
        """Benchmark hc_pre for a given T, hidden size, and dtype."""
        x, hc_fn, hc_scale, hc_base = _make_inputs(T, self.HC, d, dtype=dtype)

        # C++ kernel
        def run_cpp():
            hc_pre_cpu(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )

        # PyTorch reference
        def run_py():
            _ref_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.HC,
                self.SINKHORN_ITERS,
                self.RMS_EPS,
                self.HC_EPS,
            )

        cpp_time, cpp_std, _ = BenchmarkHelper.timeit(
            run_cpp, warmup_iters=warmup, measure_iters=measure
        )
        py_time, py_std, _ = BenchmarkHelper.timeit(
            run_py, warmup_iters=warmup, measure_iters=measure
        )

        # Throughput: input elements (x has T*hc*D elements)
        total_elements = T * self.HC * d
        throughput_cpp = total_elements / cpp_time
        throughput_py = total_elements / py_time

        BenchmarkHelper.report_benchmark(
            description, cpp_time, py_time, throughput_cpp, throughput_py
        )

    def test_benchmark_sweep_fp32(self):
        """Sweep T=1..2048 (powers of two) with fp32."""
        print()
        for d in self.HIDDEN_SIZES:
            for T in T_SWEEP:
                w, m = _bench_iters(T, d)
                self._run_benchmark(
                    T,
                    d,
                    torch.float32,
                    description=f"hc_pre (T={T:4d}, D={d}, fp32)",
                    warmup=w,
                    measure=m,
                )

    def test_benchmark_sweep_bf16(self):
        """Sweep T=1..2048 (powers of two) with bf16."""
        print()
        for d in self.HIDDEN_SIZES:
            for T in T_SWEEP:
                w, m = _bench_iters(T, d)
                self._run_benchmark(
                    T,
                    d,
                    torch.bfloat16,
                    description=f"hc_pre (T={T:4d}, D={d}, bf16)",
                    warmup=w,
                    measure=m,
                )


class BenchmarkHcPostCpu(unittest.TestCase):
    """Benchmark hc_post_cpu vs PyTorch reference."""

    HC = 4
    HIDDEN_SIZES = (4096, 7168)

    def _run_benchmark(self, T, d, dtype, description, warmup=500, measure=500):
        """Benchmark hc_post for a given T, hidden size, and dtype."""
        x, residual, post, comb = _make_post_inputs(T, self.HC, d, dtype=dtype)

        # C++ kernel
        def run_cpp():
            hc_post_cpu(x, residual, post, comb)

        # PyTorch reference
        def run_py():
            _ref_hc_post(x, residual, post, comb)

        cpp_time, cpp_std, _ = BenchmarkHelper.timeit(
            run_cpp, warmup_iters=warmup, measure_iters=measure
        )
        py_time, py_std, _ = BenchmarkHelper.timeit(
            run_py, warmup_iters=warmup, measure_iters=measure
        )

        # Throughput: input elements (x is T*D, residual is T*HC*D)
        total_elements = T * d + T * self.HC * d
        throughput_cpp = total_elements / cpp_time
        throughput_py = total_elements / py_time

        BenchmarkHelper.report_benchmark(
            description, cpp_time, py_time, throughput_cpp, throughput_py
        )

    def test_benchmark_sweep_fp32(self):
        """Sweep T=1..2048 (powers of two) with fp32."""
        print()
        for d in self.HIDDEN_SIZES:
            for T in T_SWEEP:
                w, m = _bench_iters(T, d)
                self._run_benchmark(
                    T,
                    d,
                    torch.float32,
                    description=f"hc_post (T={T:4d}, D={d}, fp32)",
                    warmup=w,
                    measure=m,
                )

    def test_benchmark_sweep_bf16(self):
        """Sweep T=1..2048 (powers of two) with bf16."""
        print()
        for d in self.HIDDEN_SIZES:
            for T in T_SWEEP:
                w, m = _bench_iters(T, d)
                self._run_benchmark(
                    T,
                    d,
                    torch.bfloat16,
                    description=f"hc_post (T={T:4d}, D={d}, bf16)",
                    warmup=w,
                    measure=m,
                )


class BenchmarkHcHeadCpu(unittest.TestCase):
    """Benchmark hc_head_cpu vs PyTorch reference."""

    HC = 4
    HIDDEN_SIZES = (4096, 7168)
    HC_EPS = 1e-6
    NORM_EPS = 1e-5

    def _make(self, T, d, dtype=torch.float32, seed=0):
        gen = torch.Generator()
        gen.manual_seed(seed)
        x = torch.randn(T, self.HC, d, dtype=dtype, generator=gen)
        hc_fn = torch.randn(self.HC, self.HC * d, dtype=dtype, generator=gen) * 0.02
        hc_scale = torch.tensor(1.0, dtype=torch.float32)
        hc_base = torch.zeros(self.HC, dtype=torch.float32)
        return x, hc_fn, hc_scale, hc_base

    def _run_benchmark(self, T, d, dtype, description, warmup=500, measure=500):
        """Benchmark hc_head for a given T, hidden size, and dtype."""
        x, hc_fn, hc_scale, hc_base = self._make(T, d, dtype=dtype)

        # C++ kernel
        def run_cpp():
            hc_head_cpu(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)

        # PyTorch reference
        def run_py():
            _ref_hc_head(x, hc_fn, hc_scale, hc_base, self.HC_EPS, self.NORM_EPS)

        cpp_time, cpp_std, _ = BenchmarkHelper.timeit(
            run_cpp, warmup_iters=warmup, measure_iters=measure
        )
        py_time, py_std, _ = BenchmarkHelper.timeit(
            run_py, warmup_iters=warmup, measure_iters=measure
        )

        # Throughput: input elements (x is T*HC*D)
        total_elements = T * self.HC * d
        throughput_cpp = total_elements / cpp_time
        throughput_py = total_elements / py_time

        BenchmarkHelper.report_benchmark(
            description, cpp_time, py_time, throughput_cpp, throughput_py
        )

    def test_benchmark_sweep_fp32(self):
        """Sweep T=1..2048 (powers of two) with fp32."""
        print()
        for d in self.HIDDEN_SIZES:
            for T in T_SWEEP:
                w, m = _bench_iters(T, d)
                self._run_benchmark(
                    T,
                    d,
                    torch.float32,
                    description=f"hc_head (T={T:4d}, D={d}, fp32)",
                    warmup=w,
                    measure=m,
                )

    def test_benchmark_sweep_bf16(self):
        """Sweep T=1..2048 (powers of two) with bf16."""
        print()
        for d in self.HIDDEN_SIZES:
            for T in T_SWEEP:
                w, m = _bench_iters(T, d)
                self._run_benchmark(
                    T,
                    d,
                    torch.bfloat16,
                    description=f"hc_head (T={T:4d}, D={d}, bf16)",
                    warmup=w,
                    measure=m,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)


"""
test_benchmark_decode_bf16 (__main__.BenchmarkHcHeadCpu)
Token generation (T=1) with bf16. ...   hc_head decode (T=1, D=4096, bf16)       | C++   0.121ms | PyTorch   0.134ms | Speedup  1.10x | Throughput (C++/PyTorch): 134969.2/122528.5 elem/ms
  hc_head decode (T=1, D=7168, bf16)       | C++   0.117ms | PyTorch   0.204ms | Speedup  1.74x | Throughput (C++/PyTorch): 244624.8/140789.4 elem/ms
ok
test_benchmark_decode_fp32 (__main__.BenchmarkHcHeadCpu)
Token generation (T=1) with fp32. ...   hc_head decode (T=1, D=4096, fp32)       | C++   0.037ms | PyTorch   0.072ms | Speedup  1.94x | Throughput (C++/PyTorch): 441826.2/228010.5 elem/ms
  hc_head decode (T=1, D=7168, fp32)       | C++   0.096ms | PyTorch   0.087ms | Speedup  0.91x | Throughput (C++/PyTorch): 299226.4/327732.5 elem/ms
ok
test_benchmark_prefill_t128_fp32 (__main__.BenchmarkHcHeadCpu)
Medium prefill (T=128) with fp32. ...   hc_head prefill (T=128, D=4096, fp32)    | C++   0.456ms | PyTorch   0.298ms | Speedup  0.65x | Throughput (C++/PyTorch): 4597611.0/7026002.6 elem/ms
  hc_head prefill (T=128, D=7168, fp32)    | C++   0.622ms | PyTorch   0.420ms | Speedup  0.67x | Throughput (C++/PyTorch): 5896848.4/8737918.0 elem/ms
ok
test_benchmark_prefill_t32_bf16 (__main__.BenchmarkHcHeadCpu)
Small prefill (T=32) with bf16. ...   hc_head prefill (T=32, D=4096, bf16)     | C++   0.194ms | PyTorch   0.382ms | Speedup  1.98x | Throughput (C++/PyTorch): 2708200.9/1371001.6 elem/ms
  hc_head prefill (T=32, D=7168, bf16)     | C++   0.252ms | PyTorch   0.332ms | Speedup  1.32x | Throughput (C++/PyTorch): 3646150.4/2760800.9 elem/ms
ok
test_benchmark_prefill_t32_fp32 (__main__.BenchmarkHcHeadCpu)
Small prefill (T=32) with fp32. ...   hc_head prefill (T=32, D=4096, fp32)     | C++   0.160ms | PyTorch   0.276ms | Speedup  1.73x | Throughput (C++/PyTorch): 3285275.0/1898271.3 elem/ms
  hc_head prefill (T=32, D=7168, fp32)     | C++   0.222ms | PyTorch   0.285ms | Speedup  1.29x | Throughput (C++/PyTorch): 4138579.2/3218251.7 elem/ms
ok
test_benchmark_prefill_t512_bf16 (__main__.BenchmarkHcHeadCpu)
Large prefill (T=512) with bf16. ...   hc_head prefill (T=512, D=4096, bf16)    | C++   0.535ms | PyTorch   1.049ms | Speedup  1.96x | Throughput (C++/PyTorch): 15686776.8/7997843.1 elem/ms
  hc_head prefill (T=512, D=7168, bf16)    | C++   0.903ms | PyTorch   2.318ms | Speedup  2.57x | Throughput (C++/PyTorch): 16260745.6/6333591.7 elem/ms
ok
test_benchmark_prefill_t512_fp32 (__main__.BenchmarkHcHeadCpu)
Large prefill (T=512) with fp32. ...   hc_head prefill (T=512, D=4096, fp32)    | C++   0.585ms | PyTorch   0.756ms | Speedup  1.29x | Throughput (C++/PyTorch): 14345128.0/11089197.1 elem/ms
  hc_head prefill (T=512, D=7168, fp32)    | C++   1.067ms | PyTorch   1.566ms | Speedup  1.47x | Throughput (C++/PyTorch): 13757299.0/9373249.4 elem/ms
ok
test_benchmark_decode_bf16 (__main__.BenchmarkHcPostCpu)
Token generation (T=1) with bf16. ...   hc_post decode (T=1, D=4096, bf16)       | C++   0.024ms | PyTorch   0.117ms | Speedup  4.95x | Throughput (C++/PyTorch): 870871.3/175782.9 elem/ms
  hc_post decode (T=1, D=7168, bf16)       | C++   0.025ms | PyTorch   0.149ms | Speedup  5.88x | Throughput (C++/PyTorch): 1416706.6/240965.3 elem/ms
ok
test_benchmark_decode_fp32 (__main__.BenchmarkHcPostCpu)
Token generation (T=1) with fp32. ...   hc_post decode (T=1, D=4096, fp32)       | C++   0.023ms | PyTorch   0.126ms | Speedup  5.56x | Throughput (C++/PyTorch): 905741.9/162879.3 elem/ms
  hc_post decode (T=1, D=7168, fp32)       | C++   0.023ms | PyTorch   0.133ms | Speedup  5.88x | Throughput (C++/PyTorch): 1580059.0/268707.5 elem/ms
ok
test_benchmark_prefill_t128_fp32 (__main__.BenchmarkHcPostCpu)
Medium prefill (T=128) with fp32. ...   hc_post prefill (T=128, D=4096, fp32)    | C++   0.046ms | PyTorch   0.313ms | Speedup  6.83x | Throughput (C++/PyTorch): 57239446.2/8376222.7 elem/ms
  hc_post prefill (T=128, D=7168, fp32)    | C++   0.072ms | PyTorch   0.612ms | Speedup  8.51x | Throughput (C++/PyTorch): 63782008.9/7497068.4 elem/ms
ok
test_benchmark_prefill_t32_bf16 (__main__.BenchmarkHcPostCpu)
Small prefill (T=32) with bf16. ...   hc_post prefill (T=32, D=4096, bf16)     | C++   0.032ms | PyTorch   0.250ms | Speedup  7.91x | Throughput (C++/PyTorch): 20779440.5/2625444.5 elem/ms
  hc_post prefill (T=32, D=7168, bf16)     | C++   0.037ms | PyTorch   0.284ms | Speedup  7.64x | Throughput (C++/PyTorch): 30914515.7/4043812.5 elem/ms
ok
test_benchmark_prefill_t32_fp32 (__main__.BenchmarkHcPostCpu)
Small prefill (T=32) with fp32. ...   hc_post prefill (T=32, D=4096, fp32)     | C++   0.028ms | PyTorch   0.195ms | Speedup  7.00x | Throughput (C++/PyTorch): 23474752.4/3352784.2 elem/ms
  hc_post prefill (T=32, D=7168, fp32)     | C++   0.032ms | PyTorch   0.223ms | Speedup  6.92x | Throughput (C++/PyTorch): 35589786.5/5140121.3 elem/ms
ok
test_benchmark_prefill_t512_bf16 (__main__.BenchmarkHcPostCpu)
Large prefill (T=512) with bf16. ...   hc_post prefill (T=512, D=4096, bf16)    | C++   0.166ms | PyTorch   1.959ms | Speedup 11.78x | Throughput (C++/PyTorch): 63033789.0/5352362.9 elem/ms
  hc_post prefill (T=512, D=7168, bf16)    | C++   0.263ms | PyTorch   6.257ms | Speedup 23.82x | Throughput (C++/PyTorch): 69850214.1/2932961.4 elem/ms
ok
test_benchmark_prefill_t512_fp32 (__main__.BenchmarkHcPostCpu)
Large prefill (T=512) with fp32. ...   hc_post prefill (T=512, D=4096, fp32)    | C++   0.207ms | PyTorch   1.868ms | Speedup  9.05x | Throughput (C++/PyTorch): 50772634.6/5612492.3 elem/ms
  hc_post prefill (T=512, D=7168, fp32)    | C++   0.471ms | PyTorch   5.962ms | Speedup 12.66x | Throughput (C++/PyTorch): 38957357.7/3077643.6 elem/ms
ok
test_benchmark_decode_bf16 (__main__.BenchmarkHcPreCpu)
Token generation (T=1) with bf16. ...   hc_pre decode (T=1, D=4096, bf16)        | C++   0.147ms | PyTorch   0.690ms | Speedup  4.70x | Throughput (C++/PyTorch): 111722.1/23757.6 elem/ms
  hc_pre decode (T=1, D=7168, bf16)        | C++   0.179ms | PyTorch   0.624ms | Speedup  3.49x | Throughput (C++/PyTorch): 160429.8/45967.7 elem/ms
ok
test_benchmark_decode_fp32 (__main__.BenchmarkHcPreCpu)
Token generation (T=1) with fp32. ...   hc_pre decode (T=1, D=4096, fp32)        | C++   0.098ms | PyTorch   0.547ms | Speedup  5.57x | Throughput (C++/PyTorch): 166683.0/29949.3 elem/ms
  hc_pre decode (T=1, D=7168, fp32)        | C++   0.098ms | PyTorch   0.470ms | Speedup  4.78x | Throughput (C++/PyTorch): 291422.0/60979.9 elem/ms
ok
test_benchmark_prefill_t128_fp32 (__main__.BenchmarkHcPreCpu)
Medium prefill (T=128) with fp32. ...   hc_pre prefill (T=128, D=4096, fp32)     | C++   0.517ms | PyTorch   1.124ms | Speedup  2.18x | Throughput (C++/PyTorch): 4059033.0/1865422.5 elem/ms
  hc_pre prefill (T=128, D=7168, fp32)     | C++   0.691ms | PyTorch   1.306ms | Speedup  1.89x | Throughput (C++/PyTorch): 5307956.1/2810765.1 elem/ms
ok
test_benchmark_prefill_t32_bf16 (__main__.BenchmarkHcPreCpu)
Small prefill (T=32) with bf16. ...   hc_pre prefill (T=32, D=4096, bf16)      | C++   0.218ms | PyTorch   0.962ms | Speedup  4.41x | Throughput (C++/PyTorch): 2406368.3/545188.0 elem/ms
  hc_pre prefill (T=32, D=7168, bf16)      | C++   0.286ms | PyTorch   1.025ms | Speedup  3.59x | Throughput (C++/PyTorch): 3213397.8/894782.3 elem/ms
ok
test_benchmark_prefill_t32_fp32 (__main__.BenchmarkHcPreCpu)
Small prefill (T=32) with fp32. ...   hc_pre prefill (T=32, D=4096, fp32)      | C++   0.165ms | PyTorch   0.756ms | Speedup  4.58x | Throughput (C++/PyTorch): 3179029.5/693414.2 elem/ms
  hc_pre prefill (T=32, D=7168, fp32)      | C++   0.228ms | PyTorch   0.820ms | Speedup  3.60x | Throughput (C++/PyTorch): 4029438.6/1118294.9 elem/ms
ok
test_benchmark_prefill_t512_bf16 (__main__.BenchmarkHcPreCpu)
Large prefill (T=512) with bf16. ...   hc_pre prefill (T=512, D=4096, bf16)     | C++   0.846ms | PyTorch   3.970ms | Speedup  4.70x | Throughput (C++/PyTorch): 9921431.6/2113070.4 elem/ms
  hc_pre prefill (T=512, D=7168, bf16)     | C++   1.235ms | PyTorch   5.091ms | Speedup  4.12x | Throughput (C++/PyTorch): 11886378.4/2883276.8 elem/ms
ok
test_benchmark_prefill_t512_fp32 (__main__.BenchmarkHcPreCpu)
Large prefill (T=512) with fp32. ...   hc_pre prefill (T=512, D=4096, fp32)     | C++   0.711ms | PyTorch   3.589ms | Speedup  5.05x | Throughput (C++/PyTorch): 11794645.0/2337329.5 elem/ms
  hc_pre prefill (T=512, D=7168, fp32)     | C++   1.250ms | PyTorch   4.007ms | Speedup  3.21x | Throughput (C++/PyTorch): 11742751.8/3663248.8 elem/ms
ok







==================================================================================================
test_benchmark_sweep_bf16 (__main__.BenchmarkHcHeadCpu)
Sweep T=1..2048 (powers of two) with bf16. ...
  hc_head (T=   1, D=4096, bf16)           | C++   0.124ms | PyTorch   0.135ms | Speedup  1.09x | Throughput (C++/PyTorch): 132042.9/121368.9 elem/ms
  hc_head (T=   2, D=4096, bf16)           | C++   0.115ms | PyTorch   0.191ms | Speedup  1.65x | Throughput (C++/PyTorch): 284080.6/171877.0 elem/ms
  hc_head (T=   4, D=4096, bf16)           | C++   0.140ms | PyTorch   0.312ms | Speedup  2.22x | Throughput (C++/PyTorch): 466520.3/210173.1 elem/ms
  hc_head (T=   8, D=4096, bf16)           | C++   0.162ms | PyTorch   0.318ms | Speedup  1.97x | Throughput (C++/PyTorch): 811385.4/411595.5 elem/ms
  hc_head (T=  16, D=4096, bf16)           | C++   0.089ms | PyTorch   0.398ms | Speedup  4.47x | Throughput (C++/PyTorch): 2946688.3/658556.3 elem/ms
  hc_head (T=  32, D=4096, bf16)           | C++   0.120ms | PyTorch   0.372ms | Speedup  3.11x | Throughput (C++/PyTorch): 4385451.4/1409002.8 elem/ms
  hc_head (T=  64, D=4096, bf16)           | C++   0.136ms | PyTorch   0.516ms | Speedup  3.78x | Throughput (C++/PyTorch): 7689539.5/2033830.9 elem/ms
  hc_head (T= 128, D=4096, bf16)           | C++   0.218ms | PyTorch   0.788ms | Speedup  3.61x | Throughput (C++/PyTorch): 9609849.0/2660473.4 elem/ms
  hc_head (T= 256, D=4096, bf16)           | C++   0.334ms | PyTorch   1.147ms | Speedup  3.43x | Throughput (C++/PyTorch): 12564041.1/3657980.2 elem/ms
  hc_head (T= 512, D=4096, bf16)           | C++   0.451ms | PyTorch   1.182ms | Speedup  2.62x | Throughput (C++/PyTorch): 18595694.0/7099931.6 elem/ms
  hc_head (T=1024, D=4096, bf16)           | C++   0.811ms | PyTorch   2.520ms | Speedup  3.11x | Throughput (C++/PyTorch): 20685296.4/6657832.1 elem/ms
  hc_head (T=2048, D=4096, bf16)           | C++   1.582ms | PyTorch   6.060ms | Speedup  3.83x | Throughput (C++/PyTorch): 21205357.4/5536850.2 elem/ms
  hc_head (T=   1, D=7168, bf16)           | C++   0.103ms | PyTorch   0.202ms | Speedup  1.96x | Throughput (C++/PyTorch): 278610.8/142064.5 elem/ms
  hc_head (T=   2, D=7168, bf16)           | C++   0.120ms | PyTorch   0.303ms | Speedup  2.52x | Throughput (C++/PyTorch): 477682.1/189227.1 elem/ms
  hc_head (T=   4, D=7168, bf16)           | C++   0.146ms | PyTorch   0.303ms | Speedup  2.08x | Throughput (C++/PyTorch): 786866.1/378293.2 elem/ms
  hc_head (T=   8, D=7168, bf16)           | C++   0.171ms | PyTorch   0.343ms | Speedup  2.01x | Throughput (C++/PyTorch): 1341723.9/668276.9 elem/ms
  hc_head (T=  16, D=7168, bf16)           | C++   0.085ms | PyTorch   0.354ms | Speedup  4.19x | Throughput (C++/PyTorch): 5426892.2/1295207.3 elem/ms
  hc_head (T=  32, D=7168, bf16)           | C++   0.101ms | PyTorch   0.311ms | Speedup  3.07x | Throughput (C++/PyTorch): 9052701.5/2948685.5 elem/ms
  hc_head (T=  64, D=7168, bf16)           | C++   0.151ms | PyTorch   0.692ms | Speedup  4.57x | Throughput (C++/PyTorch): 12116641.2/2652800.4 elem/ms
  hc_head (T= 128, D=7168, bf16)           | C++   0.303ms | PyTorch   1.131ms | Speedup  3.74x | Throughput (C++/PyTorch): 12124109.0/3245706.7 elem/ms
  hc_head (T= 256, D=7168, bf16)           | C++   0.419ms | PyTorch   1.319ms | Speedup  3.15x | Throughput (C++/PyTorch): 17515132.0/5566372.7 elem/ms
  hc_head (T= 512, D=7168, bf16)           | C++   0.747ms | PyTorch   2.927ms | Speedup  3.92x | Throughput (C++/PyTorch): 19655078.7/5015944.2 elem/ms
  hc_head (T=1024, D=7168, bf16)           | C++   1.436ms | PyTorch   4.635ms | Speedup  3.23x | Throughput (C++/PyTorch): 20443796.8/6333827.2 elem/ms
  hc_head (T=2048, D=7168, bf16)           | C++   2.751ms | PyTorch  13.643ms | Speedup  4.96x | Throughput (C++/PyTorch): 21343244.6/4304007.1 elem/ms
ok
test_benchmark_sweep_fp32 (__main__.BenchmarkHcHeadCpu)
Sweep T=1..2048 (powers of two) with fp32. ...
  hc_head (T=   1, D=4096, fp32)           | C++   0.055ms | PyTorch   0.073ms | Speedup  1.33x | Throughput (C++/PyTorch): 298026.0/223473.9 elem/ms
  hc_head (T=   2, D=4096, fp32)           | C++   0.084ms | PyTorch   0.147ms | Speedup  1.76x | Throughput (C++/PyTorch): 391809.3/223115.6 elem/ms
  hc_head (T=   4, D=4096, fp32)           | C++   0.089ms | PyTorch   0.185ms | Speedup  2.08x | Throughput (C++/PyTorch): 738993.0/354879.8 elem/ms
  hc_head (T=   8, D=4096, fp32)           | C++   0.109ms | PyTorch   0.231ms | Speedup  2.13x | Throughput (C++/PyTorch): 1207833.5/567742.7 elem/ms
  hc_head (T=  16, D=4096, fp32)           | C++   0.027ms | PyTorch   0.275ms | Speedup 10.01x | Throughput (C++/PyTorch): 9544883.1/953140.1 elem/ms
  hc_head (T=  32, D=4096, fp32)           | C++   0.036ms | PyTorch   0.276ms | Speedup  7.58x | Throughput (C++/PyTorch): 14394310.1/1898842.6 elem/ms
  hc_head (T=  64, D=4096, fp32)           | C++   0.040ms | PyTorch   0.238ms | Speedup  5.98x | Throughput (C++/PyTorch): 26289488.1/4396842.9 elem/ms
  hc_head (T= 128, D=4096, fp32)           | C++   0.052ms | PyTorch   0.302ms | Speedup  5.77x | Throughput (C++/PyTorch): 40033534.0/6943435.7 elem/ms
  hc_head (T= 256, D=4096, fp32)           | C++   0.079ms | PyTorch   0.389ms | Speedup  4.94x | Throughput (C++/PyTorch): 53299156.4/10790089.8 elem/ms
  hc_head (T= 512, D=4096, fp32)           | C++   0.229ms | PyTorch   0.783ms | Speedup  3.42x | Throughput (C++/PyTorch): 36640333.0/10708476.3 elem/ms
  hc_head (T=1024, D=4096, fp32)           | C++   0.889ms | PyTorch   2.034ms | Speedup  2.29x | Throughput (C++/PyTorch): 18882150.4/8247866.8 elem/ms
  hc_head (T=2048, D=4096, fp32)           | C++   1.272ms | PyTorch   4.474ms | Speedup  3.52x | Throughput (C++/PyTorch): 26385943.2/7499647.5 elem/ms
  hc_head (T=   1, D=7168, fp32)           | C++   0.048ms | PyTorch   0.088ms | Speedup  1.83x | Throughput (C++/PyTorch): 598483.9/327354.5 elem/ms
  hc_head (T=   2, D=7168, fp32)           | C++   0.081ms | PyTorch   0.162ms | Speedup  2.01x | Throughput (C++/PyTorch): 710360.9/353281.0 elem/ms
  hc_head (T=   4, D=7168, fp32)           | C++   0.121ms | PyTorch   0.188ms | Speedup  1.55x | Throughput (C++/PyTorch): 948339.7/610099.9 elem/ms
  hc_head (T=   8, D=7168, fp32)           | C++   0.137ms | PyTorch   0.302ms | Speedup  2.20x | Throughput (C++/PyTorch): 1670343.2/758763.8 elem/ms
  hc_head (T=  16, D=7168, fp32)           | C++   0.040ms | PyTorch   0.309ms | Speedup  7.66x | Throughput (C++/PyTorch): 11371004.1/1485305.0 elem/ms
  hc_head (T=  32, D=7168, fp32)           | C++   0.041ms | PyTorch   0.284ms | Speedup  6.92x | Throughput (C++/PyTorch): 22320981.7/3227474.0 elem/ms
  hc_head (T=  64, D=7168, fp32)           | C++   0.054ms | PyTorch   0.269ms | Speedup  5.02x | Throughput (C++/PyTorch): 34157831.7/6810189.5 elem/ms
  hc_head (T= 128, D=7168, fp32)           | C++   0.073ms | PyTorch   0.402ms | Speedup  5.51x | Throughput (C++/PyTorch): 50340388.9/9138426.6 elem/ms
  hc_head (T= 256, D=7168, fp32)           | C++   0.195ms | PyTorch   0.614ms | Speedup  3.15x | Throughput (C++/PyTorch): 37687711.0/11961430.3 elem/ms
  hc_head (T= 512, D=7168, fp32)           | C++   0.585ms | PyTorch   2.207ms | Speedup  3.77x | Throughput (C++/PyTorch): 25093518.0/6651899.9 elem/ms
  hc_head (T=1024, D=7168, fp32)           | C++   1.162ms | PyTorch   3.533ms | Speedup  3.04x | Throughput (C++/PyTorch): 25262987.8/8310285.9 elem/ms
  hc_head (T=2048, D=7168, fp32)           | C++   2.736ms | PyTorch  12.418ms | Speedup  4.54x | Throughput (C++/PyTorch): 21460416.5/4728543.8 elem/ms
ok
test_benchmark_sweep_bf16 (__main__.BenchmarkHcPostCpu)
Sweep T=1..2048 (powers of two) with bf16. ...
  hc_post (T=   1, D=4096, bf16)           | C++   0.021ms | PyTorch   0.116ms | Speedup  5.50x | Throughput (C++/PyTorch): 971800.0/176589.0 elem/ms
  hc_post (T=   2, D=4096, bf16)           | C++   0.023ms | PyTorch   0.159ms | Speedup  6.78x | Throughput (C++/PyTorch): 1746603.9/257657.8 elem/ms
  hc_post (T=   4, D=4096, bf16)           | C++   0.027ms | PyTorch   0.218ms | Speedup  8.23x | Throughput (C++/PyTorch): 3087329.9/375160.0 elem/ms
  hc_post (T=   8, D=4096, bf16)           | C++   0.027ms | PyTorch   0.268ms | Speedup  9.80x | Throughput (C++/PyTorch): 5987080.6/611211.5 elem/ms
  hc_post (T=  16, D=4096, bf16)           | C++   0.030ms | PyTorch   0.397ms | Speedup 13.31x | Throughput (C++/PyTorch): 10982870.4/824888.0 elem/ms
  hc_post (T=  32, D=4096, bf16)           | C++   0.032ms | PyTorch   0.262ms | Speedup  8.29x | Throughput (C++/PyTorch): 20716504.2/2499647.9 elem/ms
  hc_post (T=  64, D=4096, bf16)           | C++   0.042ms | PyTorch   0.230ms | Speedup  5.41x | Throughput (C++/PyTorch): 30845734.5/5704227.7 elem/ms
  hc_post (T= 128, D=4096, bf16)           | C++   0.096ms | PyTorch   0.407ms | Speedup  4.24x | Throughput (C++/PyTorch): 27316041.1/6439593.1 elem/ms
  hc_post (T= 256, D=4096, bf16)           | C++   0.161ms | PyTorch   1.105ms | Speedup  6.85x | Throughput (C++/PyTorch): 32494719.2/4743254.7 elem/ms
  hc_post (T= 512, D=4096, bf16)           | C++   0.299ms | PyTorch   2.149ms | Speedup  7.19x | Throughput (C++/PyTorch): 35073487.3/4878872.5 elem/ms
  hc_post (T=1024, D=4096, bf16)           | C++   0.595ms | PyTorch   6.787ms | Speedup 11.41x | Throughput (C++/PyTorch): 35262732.5/3089998.0 elem/ms
  hc_post (T=2048, D=4096, bf16)           | C++   1.210ms | PyTorch  22.143ms | Speedup 18.30x | Throughput (C++/PyTorch): 34671120.3/1894156.5 elem/ms
  hc_post (T=   1, D=7168, bf16)           | C++   0.020ms | PyTorch   0.147ms | Speedup  7.45x | Throughput (C++/PyTorch): 1821570.9/244512.4 elem/ms
  hc_post (T=   2, D=7168, bf16)           | C++   0.024ms | PyTorch   0.199ms | Speedup  8.17x | Throughput (C++/PyTorch): 2943768.0/360434.4 elem/ms
  hc_post (T=   4, D=7168, bf16)           | C++   0.027ms | PyTorch   0.224ms | Speedup  8.39x | Throughput (C++/PyTorch): 5362372.7/638810.6 elem/ms
  hc_post (T=   8, D=7168, bf16)           | C++   0.029ms | PyTorch   0.263ms | Speedup  9.13x | Throughput (C++/PyTorch): 9968051.0/1091992.9 elem/ms
  hc_post (T=  16, D=7168, bf16)           | C++   0.032ms | PyTorch   0.410ms | Speedup 12.86x | Throughput (C++/PyTorch): 17977678.8/1397560.8 elem/ms
  hc_post (T=  32, D=7168, bf16)           | C++   0.039ms | PyTorch   0.273ms | Speedup  7.04x | Throughput (C++/PyTorch): 29592163.4/4203399.7 elem/ms
  hc_post (T=  64, D=7168, bf16)           | C++   0.088ms | PyTorch   0.379ms | Speedup  4.29x | Throughput (C++/PyTorch): 25948992.3/6044717.3 elem/ms
  hc_post (T= 128, D=7168, bf16)           | C++   0.153ms | PyTorch   1.039ms | Speedup  6.77x | Throughput (C++/PyTorch): 29908365.5/4416781.2 elem/ms
  hc_post (T= 256, D=7168, bf16)           | C++   0.270ms | PyTorch   1.898ms | Speedup  7.02x | Throughput (C++/PyTorch): 33935815.3/4832906.0 elem/ms
  hc_post (T= 512, D=7168, bf16)           | C++   0.526ms | PyTorch   5.426ms | Speedup 10.31x | Throughput (C++/PyTorch): 34886096.0/3382189.0 elem/ms
  hc_post (T=1024, D=7168, bf16)           | C++   1.045ms | PyTorch  18.303ms | Speedup 17.52x | Throughput (C++/PyTorch): 35126037.8/2005096.1 elem/ms
  hc_post (T=2048, D=7168, bf16)           | C++   2.356ms | PyTorch  37.405ms | Speedup 15.88x | Throughput (C++/PyTorch): 31160534.4/1962314.3 elem/ms
ok
test_benchmark_sweep_fp32 (__main__.BenchmarkHcPostCpu)
Sweep T=1..2048 (powers of two) with fp32. ...
  hc_post (T=   1, D=4096, fp32)           | C++   0.021ms | PyTorch   0.124ms | Speedup  5.81x | Throughput (C++/PyTorch): 962493.0/165668.1 elem/ms
  hc_post (T=   2, D=4096, fp32)           | C++   0.022ms | PyTorch   0.128ms | Speedup  5.92x | Throughput (C++/PyTorch): 1888826.9/318996.7 elem/ms
  hc_post (T=   4, D=4096, fp32)           | C++   0.023ms | PyTorch   0.157ms | Speedup  6.96x | Throughput (C++/PyTorch): 3624304.6/520374.6 elem/ms
  hc_post (T=   8, D=4096, fp32)           | C++   0.024ms | PyTorch   0.209ms | Speedup  8.59x | Throughput (C++/PyTorch): 6743013.9/785140.6 elem/ms
  hc_post (T=  16, D=4096, fp32)           | C++   0.026ms | PyTorch   0.318ms | Speedup 12.12x | Throughput (C++/PyTorch): 12487080.5/1030091.4 elem/ms
  hc_post (T=  32, D=4096, fp32)           | C++   0.029ms | PyTorch   0.166ms | Speedup  5.81x | Throughput (C++/PyTorch): 22926463.3/3947287.1 elem/ms
  hc_post (T=  64, D=4096, fp32)           | C++   0.035ms | PyTorch   0.147ms | Speedup  4.18x | Throughput (C++/PyTorch): 37329552.1/8924279.1 elem/ms
  hc_post (T= 128, D=4096, fp32)           | C++   0.049ms | PyTorch   0.232ms | Speedup  4.71x | Throughput (C++/PyTorch): 53271557.4/11312277.2 elem/ms
  hc_post (T= 256, D=4096, fp32)           | C++   0.085ms | PyTorch   0.703ms | Speedup  8.29x | Throughput (C++/PyTorch): 61823608.6/7453844.7 elem/ms
  hc_post (T= 512, D=4096, fp32)           | C++   0.256ms | PyTorch   1.651ms | Speedup  6.45x | Throughput (C++/PyTorch): 40998347.8/6351686.0 elem/ms
  hc_post (T=1024, D=4096, fp32)           | C++   0.739ms | PyTorch   5.749ms | Speedup  7.78x | Throughput (C++/PyTorch): 28382783.9/3647998.7 elem/ms
  hc_post (T=2048, D=4096, fp32)           | C++   1.662ms | PyTorch  16.558ms | Speedup  9.97x | Throughput (C++/PyTorch): 25243702.0/2533143.2 elem/ms
  hc_post (T=   1, D=7168, fp32)           | C++   0.015ms | PyTorch   0.133ms | Speedup  8.94x | Throughput (C++/PyTorch): 2413871.6/269954.7 elem/ms
  hc_post (T=   2, D=7168, fp32)           | C++   0.023ms | PyTorch   0.142ms | Speedup  6.30x | Throughput (C++/PyTorch): 3184316.2/505823.4 elem/ms
  hc_post (T=   4, D=7168, fp32)           | C++   0.025ms | PyTorch   0.163ms | Speedup  6.58x | Throughput (C++/PyTorch): 5783033.5/878392.8 elem/ms
  hc_post (T=   8, D=7168, fp32)           | C++   0.026ms | PyTorch   0.195ms | Speedup  7.53x | Throughput (C++/PyTorch): 11096482.6/1473161.2 elem/ms
  hc_post (T=  16, D=7168, fp32)           | C++   0.029ms | PyTorch   0.454ms | Speedup 15.82x | Throughput (C++/PyTorch): 20003252.3/1264152.8 elem/ms
  hc_post (T=  32, D=7168, fp32)           | C++   0.034ms | PyTorch   0.215ms | Speedup  6.37x | Throughput (C++/PyTorch): 33995275.6/5336877.2 elem/ms
  hc_post (T=  64, D=7168, fp32)           | C++   0.047ms | PyTorch   0.236ms | Speedup  5.02x | Throughput (C++/PyTorch): 48816778.2/9729473.4 elem/ms
  hc_post (T= 128, D=7168, fp32)           | C++   0.066ms | PyTorch   0.608ms | Speedup  9.27x | Throughput (C++/PyTorch): 69981248.0/7546136.0 elem/ms
  hc_post (T= 256, D=7168, fp32)           | C++   0.188ms | PyTorch   1.352ms | Speedup  7.20x | Throughput (C++/PyTorch): 48859059.7/6786412.4 elem/ms
  hc_post (T= 512, D=7168, fp32)           | C++   0.673ms | PyTorch   4.976ms | Speedup  7.40x | Throughput (C++/PyTorch): 27278173.5/3687353.6 elem/ms
  hc_post (T=1024, D=7168, fp32)           | C++   1.333ms | PyTorch  14.464ms | Speedup 10.85x | Throughput (C++/PyTorch): 27531923.2/2537408.9 elem/ms
  hc_post (T=2048, D=7168, fp32)           | C++   3.350ms | PyTorch  36.275ms | Speedup 10.83x | Throughput (C++/PyTorch): 21910360.0/2023439.0 elem/ms
ok
test_benchmark_sweep_bf16 (__main__.BenchmarkHcPreCpu)
Sweep T=1..2048 (powers of two) with bf16. ...
  hc_pre (T=   1, D=4096, bf16)            | C++   0.154ms | PyTorch   0.725ms | Speedup  4.72x | Throughput (C++/PyTorch): 106733.6/22607.3 elem/ms
  hc_pre (T=   2, D=4096, bf16)            | C++   0.133ms | PyTorch   0.772ms | Speedup  5.80x | Throughput (C++/PyTorch): 246125.5/42429.5 elem/ms
  hc_pre (T=   4, D=4096, bf16)            | C++   0.170ms | PyTorch   0.896ms | Speedup  5.28x | Throughput (C++/PyTorch): 386293.1/73156.1 elem/ms
  hc_pre (T=   8, D=4096, bf16)            | C++   0.216ms | PyTorch   1.019ms | Speedup  4.71x | Throughput (C++/PyTorch): 605610.8/128600.2 elem/ms
  hc_pre (T=  16, D=4096, bf16)            | C++   0.181ms | PyTorch   0.863ms | Speedup  4.77x | Throughput (C++/PyTorch): 1448326.9/303896.3 elem/ms
  hc_pre (T=  32, D=4096, bf16)            | C++   0.220ms | PyTorch   0.976ms | Speedup  4.44x | Throughput (C++/PyTorch): 2386401.9/537039.5 elem/ms
  hc_pre (T=  64, D=4096, bf16)            | C++   0.316ms | PyTorch   1.088ms | Speedup  3.45x | Throughput (C++/PyTorch): 3322843.4/963910.3 elem/ms
  hc_pre (T= 128, D=4096, bf16)            | C++   0.561ms | PyTorch   1.558ms | Speedup  2.78x | Throughput (C++/PyTorch): 3735733.4/1346182.8 elem/ms
  hc_pre (T= 256, D=4096, bf16)            | C++   1.363ms | PyTorch   3.018ms | Speedup  2.21x | Throughput (C++/PyTorch): 3077773.2/1389710.6 elem/ms
  hc_pre (T= 512, D=4096, bf16)            | C++   1.270ms | PyTorch   3.868ms | Speedup  3.05x | Throughput (C++/PyTorch): 6607541.6/2168890.7 elem/ms
  hc_pre (T=1024, D=4096, bf16)            | C++   2.806ms | PyTorch   8.068ms | Speedup  2.87x | Throughput (C++/PyTorch): 5978504.4/2079600.1 elem/ms
  hc_pre (T=2048, D=4096, bf16)            | C++   3.794ms | PyTorch  12.234ms | Speedup  3.22x | Throughput (C++/PyTorch): 8843970.1/2742789.7 elem/ms
  hc_pre (T=   1, D=7168, bf16)            | C++   0.167ms | PyTorch   0.635ms | Speedup  3.79x | Throughput (C++/PyTorch): 171193.2/45154.0 elem/ms
  hc_pre (T=   2, D=7168, bf16)            | C++   0.186ms | PyTorch   0.905ms | Speedup  4.87x | Throughput (C++/PyTorch): 308410.3/63372.6 elem/ms
  hc_pre (T=   4, D=7168, bf16)            | C++   0.300ms | PyTorch   1.008ms | Speedup  3.36x | Throughput (C++/PyTorch): 382754.0/113830.7 elem/ms
  hc_pre (T=   8, D=7168, bf16)            | C++   0.404ms | PyTorch   1.062ms | Speedup  2.63x | Throughput (C++/PyTorch): 568090.0/216085.1 elem/ms
  hc_pre (T=  16, D=7168, bf16)            | C++   0.194ms | PyTorch   0.918ms | Speedup  4.74x | Throughput (C++/PyTorch): 2368977.3/499830.2 elem/ms
  hc_pre (T=  32, D=7168, bf16)            | C++   0.285ms | PyTorch   1.020ms | Speedup  3.58x | Throughput (C++/PyTorch): 3217480.1/899932.4 elem/ms
  hc_pre (T=  64, D=7168, bf16)            | C++   0.461ms | PyTorch   1.279ms | Speedup  2.78x | Throughput (C++/PyTorch): 3981534.9/1434747.4 elem/ms
  hc_pre (T= 128, D=7168, bf16)            | C++   1.058ms | PyTorch   1.893ms | Speedup  1.79x | Throughput (C++/PyTorch): 3468658.7/1938379.7 elem/ms
  hc_pre (T= 256, D=7168, bf16)            | C++   1.590ms | PyTorch   3.032ms | Speedup  1.91x | Throughput (C++/PyTorch): 4615247.5/2420490.3 elem/ms
  hc_pre (T= 512, D=7168, bf16)            | C++   2.174ms | PyTorch   5.292ms | Speedup  2.43x | Throughput (C++/PyTorch): 6751187.1/2774002.2 elem/ms
  hc_pre (T=1024, D=7168, bf16)            | C++   3.923ms | PyTorch  11.189ms | Speedup  2.85x | Throughput (C++/PyTorch): 7483216.2/2624045.2 elem/ms
  hc_pre (T=2048, D=7168, bf16)            | C++   5.511ms | PyTorch  21.111ms | Speedup  3.83x | Throughput (C++/PyTorch): 10654878.9/2781460.6 elem/ms
ok
test_benchmark_sweep_fp32 (__main__.BenchmarkHcPreCpu)
Sweep T=1..2048 (powers of two) with fp32. ...
  hc_pre (T=   1, D=4096, fp32)            | C++   0.125ms | PyTorch   0.546ms | Speedup  4.38x | Throughput (C++/PyTorch): 131391.2/30009.4 elem/ms
  hc_pre (T=   2, D=4096, fp32)            | C++   0.115ms | PyTorch   0.592ms | Speedup  5.16x | Throughput (C++/PyTorch): 285328.3/55329.7 elem/ms
  hc_pre (T=   4, D=4096, fp32)            | C++   0.146ms | PyTorch   0.686ms | Speedup  4.69x | Throughput (C++/PyTorch): 448008.1/95468.1 elem/ms
  hc_pre (T=   8, D=4096, fp32)            | C++   0.210ms | PyTorch   0.776ms | Speedup  3.69x | Throughput (C++/PyTorch): 623403.8/168976.4 elem/ms
  hc_pre (T=  16, D=4096, fp32)            | C++   0.119ms | PyTorch   0.708ms | Speedup  5.96x | Throughput (C++/PyTorch): 2205693.1/370287.4 elem/ms
  hc_pre (T=  32, D=4096, fp32)            | C++   0.166ms | PyTorch   0.765ms | Speedup  4.60x | Throughput (C++/PyTorch): 3151901.4/685242.2 elem/ms
  hc_pre (T=  64, D=4096, fp32)            | C++   0.268ms | PyTorch   0.835ms | Speedup  3.12x | Throughput (C++/PyTorch): 3918794.5/1255816.4 elem/ms
  hc_pre (T= 128, D=4096, fp32)            | C++   0.508ms | PyTorch   1.135ms | Speedup  2.24x | Throughput (C++/PyTorch): 4131161.8/1847121.6 elem/ms
  hc_pre (T= 256, D=4096, fp32)            | C++   0.781ms | PyTorch   2.556ms | Speedup  3.27x | Throughput (C++/PyTorch): 5373303.6/1640875.3 elem/ms
  hc_pre (T= 512, D=4096, fp32)            | C++   0.748ms | PyTorch   3.603ms | Speedup  4.81x | Throughput (C++/PyTorch): 11207919.0/2328198.7 elem/ms
  hc_pre (T=1024, D=4096, fp32)            | C++   2.637ms | PyTorch   7.002ms | Speedup  2.66x | Throughput (C++/PyTorch): 6363104.2/2396105.7 elem/ms
  hc_pre (T=2048, D=4096, fp32)            | C++   3.978ms | PyTorch   9.258ms | Speedup  2.33x | Throughput (C++/PyTorch): 8434932.9/3624356.7 elem/ms
  hc_pre (T=   1, D=7168, fp32)            | C++   0.054ms | PyTorch   0.473ms | Speedup  8.69x | Throughput (C++/PyTorch): 527117.3/60673.5 elem/ms
  hc_pre (T=   2, D=7168, fp32)            | C++   0.134ms | PyTorch   0.752ms | Speedup  5.60x | Throughput (C++/PyTorch): 427048.9/76260.9 elem/ms
  hc_pre (T=   4, D=7168, fp32)            | C++   0.203ms | PyTorch   0.900ms | Speedup  4.43x | Throughput (C++/PyTorch): 564347.1/127402.5 elem/ms
  hc_pre (T=   8, D=7168, fp32)            | C++   0.301ms | PyTorch   0.936ms | Speedup  3.11x | Throughput (C++/PyTorch): 761571.3/245002.6 elem/ms
  hc_pre (T=  16, D=7168, fp32)            | C++   0.153ms | PyTorch   0.749ms | Speedup  4.91x | Throughput (C++/PyTorch): 3008106.4/612166.4 elem/ms
  hc_pre (T=  32, D=7168, fp32)            | C++   0.231ms | PyTorch   0.831ms | Speedup  3.60x | Throughput (C++/PyTorch): 3974561.3/1104126.4 elem/ms
  hc_pre (T=  64, D=7168, fp32)            | C++   0.400ms | PyTorch   0.888ms | Speedup  2.22x | Throughput (C++/PyTorch): 4591848.3/2065789.0 elem/ms
  hc_pre (T= 128, D=7168, fp32)            | C++   0.733ms | PyTorch   1.306ms | Speedup  1.78x | Throughput (C++/PyTorch): 5003653.4/2810498.2 elem/ms
  hc_pre (T= 256, D=7168, fp32)            | C++   0.847ms | PyTorch   2.630ms | Speedup  3.11x | Throughput (C++/PyTorch): 8667393.8/2790728.6 elem/ms
  hc_pre (T= 512, D=7168, fp32)            | C++   1.784ms | PyTorch   4.231ms | Speedup  2.37x | Throughput (C++/PyTorch): 8228641.8/3469306.4 elem/ms
  hc_pre (T=1024, D=7168, fp32)            | C++   3.860ms | PyTorch   8.206ms | Speedup  2.13x | Throughput (C++/PyTorch): 7605679.4/3577926.8 elem/ms
  hc_pre (T=2048, D=7168, fp32)            | C++   5.965ms | PyTorch  16.910ms | Speedup  2.84x | Throughput (C++/PyTorch): 9844759.9/3472517.3 elem/ms
ok
test_benchmark_sweep (__main__.BenchmarkHcSplitSinkhornCpu)
Sweep T=1..2048 (powers of two). ...
  hc_split_sinkhorn (T=   1)               | C++   0.403ms | PyTorch   0.406ms | Speedup  1.01x | Throughput (C++/PyTorch):   119.2/  118.4 elem/ms
  hc_split_sinkhorn (T=   2)               | C++   0.407ms | PyTorch   0.407ms | Speedup  1.00x | Throughput (C++/PyTorch):   236.0/  235.6 elem/ms
  hc_split_sinkhorn (T=   4)               | C++   0.414ms | PyTorch   0.417ms | Speedup  1.01x | Throughput (C++/PyTorch):   463.4/  460.3 elem/ms
  hc_split_sinkhorn (T=   8)               | C++   0.428ms | PyTorch   0.428ms | Speedup  1.00x | Throughput (C++/PyTorch):   898.0/  896.3 elem/ms
  hc_split_sinkhorn (T=  16)               | C++   0.452ms | PyTorch   0.451ms | Speedup  1.00x | Throughput (C++/PyTorch):  1698.9/ 1703.3 elem/ms
  hc_split_sinkhorn (T=  32)               | C++   0.497ms | PyTorch   0.496ms | Speedup  1.00x | Throughput (C++/PyTorch):  3088.5/ 3098.0 elem/ms
  hc_split_sinkhorn (T=  64)               | C++   0.583ms | PyTorch   0.580ms | Speedup  0.99x | Throughput (C++/PyTorch):  5266.3/ 5295.7 elem/ms
  hc_split_sinkhorn (T= 128)               | C++   0.756ms | PyTorch   0.756ms | Speedup  1.00x | Throughput (C++/PyTorch):  8132.1/ 8132.1 elem/ms
  hc_split_sinkhorn (T= 256)               | C++   1.903ms | PyTorch   1.907ms | Speedup  1.00x | Throughput (C++/PyTorch):  6457.4/ 6442.6 elem/ms
  hc_split_sinkhorn (T= 512)               | C++   2.544ms | PyTorch   2.572ms | Speedup  1.01x | Throughput (C++/PyTorch):  9659.4/ 9554.3 elem/ms
  hc_split_sinkhorn (T=1024)               | C++   3.890ms | PyTorch   3.894ms | Speedup  1.00x | Throughput (C++/PyTorch): 12634.3/12622.0 elem/ms
  hc_split_sinkhorn (T=2048)               | C++   4.980ms | PyTorch   4.997ms | Speedup  1.00x | Throughput (C++/PyTorch): 19739.8/19671.8 elem/ms
ok
"""
