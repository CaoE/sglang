"""CPU-optimized MHC (Multi-Head Channel) kernels for Intel Xeon.

These pure-PyTorch implementations are used when running with
SGLANG_USE_CPU_ENGINE=1, and serve as the fallback path on any non-CUDA device.
The math exactly mirrors the TileLang CUDA kernels in mhc.py so the two paths
produce numerically comparable results.

Sinkhorn reference (mirrors hc_split_sinkhorn_kernel in mhc.py):
  pre[t, j]    = sigmoid(mixes[t, j]              * scale[0] + base[j])           + eps
  post[t, j]   = 2 * sigmoid(mixes[t, j+hc]       * scale[1] + base[j+hc])
  comb[t, j, k] = mixes[t, j*hc + k + 2*hc]       * scale[2] + base[j*hc+k+2*hc]
  → comb is made doubly-stochastic via alternating row / column normalization.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional C++ kernels (compiled via sgl-kernel for Xeon CPU)
# ---------------------------------------------------------------------------
try:
    from torch.ops import sgl_kernel as _sgl_ops

    _HAS_CPP_MHC = all(
        hasattr(_sgl_ops, name)
        for name in (
            "hc_pre_fused_cpu",
            "hc_post_fused_cpu",
            "hc_head_fused_cpu",
        )
    )
    # print("C++ MHC kernels available:", _HAS_CPP_MHC, flush=True)
except Exception:
    _HAS_CPP_MHC = False


# ---------------------------------------------------------------------------
# hc_split_sinkhorn_cpu
# ---------------------------------------------------------------------------


def hc_split_sinkhorn_cpu(
    mixes: torch.Tensor,  # [T, mix_hc]  float32, mix_hc = (2+hc)*hc
    hc_scale: torch.Tensor,  # [3]          float32
    hc_base: torch.Tensor,  # [mix_hc]     float32
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch implementation of ``hc_split_sinkhorn_kernel`` (mhc.py).

    Dispatches to the compiled C++ kernel when sgl-kernel is installed and
    hc_mult==4; falls back to pure-PyTorch otherwise.

    Returns:
        pre:  [T, hc]       – sigmoid gates for pre-layer stream mixing
        post: [T, hc]       – 2*sigmoid gates for post-layer scaling
        comb: [T, hc, hc]   – doubly-stochastic residual routing matrix
    """
    return _hc_split_sinkhorn_torch(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
    )


def _hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch fallback for hc_split_sinkhorn_cpu."""
    hc = hc_mult

    # ---- pre: sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps ----------
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps  # [T, hc]

    # ---- post: 2 * sigmoid(mixes[:, hc:2hc] * scale[1] + base[hc:2hc]) ----
    post = 2.0 * torch.sigmoid(
        mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )  # [T, hc]

    # ---- comb logits: [T, hc, hc] -------------------------------------------
    comb = mixes[:, 2 * hc :].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc :].view(
        hc, hc
    )  # [T, hc, hc]

    # ---- Sinkhorn normalization (row-stable softmax first) ------------------
    comb = comb - comb.amax(dim=-1, keepdim=True)
    comb = comb.exp()  # [T, hc, hc]

    # First iteration (mirrors TileLang kernel exactly)
    row_sum = comb.sum(dim=-1, keepdim=True)
    comb = comb / row_sum + eps

    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + eps)

    # Remaining (sinkhorn_iters - 1) iterations
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + eps)

        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + eps)

    return pre, post, comb


# ---------------------------------------------------------------------------
# hc_pre_cpu
# ---------------------------------------------------------------------------


def hc_pre_cpu(
    x: torch.Tensor,  # [T, hc, d]
    hc_fn: torch.Tensor,  # [mix_hc, hc*d]
    hc_scale: torch.Tensor,  # [3]
    hc_base: torch.Tensor,  # [mix_hc]
    hc_mult: int,
    sinkhorn_iters: int,
    rms_norm_eps: float,
    hc_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU implementation of the fused mhc_pre operation.

    Computes RMS-normalised linear projections of the stacked hc residual
    streams, derives per-token gating coefficients via Sinkhorn, and returns
    the collapsed single-stream layer input together with the coefficients
    needed by the subsequent ``hc_post_cpu`` call.

    Returns:
        y:    [T, d]       – weighted combination of hc input streams
        post: [T, hc]      – post-layer mixing gates  (passed to hc_post)
        comb: [T, hc, hc]  – residual routing matrix  (passed to hc_post)
    """
    dtype = x.dtype
    if _HAS_CPP_MHC and hc_mult == 4:
        # print("Using C++ hc_pre_fused_cpu kernel", flush=True)
        y, post, comb = torch.ops.sgl_kernel.hc_pre_fused_cpu(
            x.contiguous(),
            hc_fn.float().contiguous(),
            hc_scale.float(),
            hc_base.float(),
            hc_mult,
            sinkhorn_iters,
            rms_norm_eps,
            hc_eps,
        )
        return y, post, comb

    dtype = x.dtype
    # x: [T, hc, d]
    # Flatten along hc and hidden dims for the projection
    x_flat = x.flatten(1).float()  # [T, hc*d]
    rms = x_flat.square().mean(dim=-1, keepdim=True)  # [T, 1]
    rsqrt = torch.rsqrt(rms + rms_norm_eps)  # [T, 1]

    # Linear projection: [T, mix_hc]  (uses MKL SGEMM on Intel)
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt  # [T, mix_hc]

    # Gating coefficients
    pre, post, comb = hc_split_sinkhorn_cpu(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
    )  # pre: [T, hc], post: [T, hc], comb: [T, hc, hc]

    # Weighted sum over hc channels → single-stream layer input
    # y[t, k] = sum_j pre[t, j] * x[t, j, k]
    y = (pre.unsqueeze(-1) * x.float()).sum(dim=1)  # [T, d]

    return y.to(dtype), post, comb


# ---------------------------------------------------------------------------
# hc_post_cpu
# ---------------------------------------------------------------------------


def hc_post_cpu(
    x: torch.Tensor,  # [T, d]      – sublayer (attn / MLP) output
    residual: torch.Tensor,  # [T, hc, d]  – pre-layer residual streams
    post: torch.Tensor,  # [T, hc]     – from hc_pre_cpu
    comb: torch.Tensor,  # [T, hc, hc] – from hc_pre_cpu
) -> torch.Tensor:
    """CPU implementation of the fused mhc_post operation.

    Combines the sublayer output back into the hc residual streams:
        out[t, j, k] = post[t, j] * x[t, k]  +  sum_i comb[t, i, j] * residual[t, i, k]

    This mirrors the TileLang mhc_post kernel semantics exactly.

    Returns:
        out: [T, hc, d]
    """
    if _HAS_CPP_MHC:
        hc = residual.size(1)
        if hc == 4:
            return torch.ops.sgl_kernel.hc_post_fused_cpu(
                x.contiguous(),
                residual.contiguous(),
                post.float(),
                comb.float(),
            )

    # Broadcast post-scaled sublayer output to [T, hc, d]
    out = post.float().unsqueeze(-1) * x.float().unsqueeze(1)  # [T, hc, d]

    # Add residual contribution via batched matmul.
    # comb^T[t, j, i] == comb[t, i, j], so:
    #   bmm(comb^T, residual)[t, j, k] = sum_i comb[t, i, j] * residual[t, i, k]
    out = out + torch.bmm(
        comb.float().permute(0, 2, 1).contiguous(),  # [T, hc_j, hc_i]
        residual.float(),  # [T, hc_i, d]
    )  # [T, hc, d]

    return out.type_as(x)


# ---------------------------------------------------------------------------
# hc_head_cpu
# ---------------------------------------------------------------------------


def hc_head_cpu(
    x: torch.Tensor,  # [T, hc, d]
    hc_fn: torch.Tensor,  # [hc, hc*d]
    hc_scale: torch.Tensor,  # scalar or [1]
    hc_base: torch.Tensor,  # [hc]
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    """CPU implementation of hc_head (model output collapsing).

    Collapses hc residual streams into a single output hidden state using
    RMS-normalised sigmoid gates.

    Returns: [T, d]
    """
    hc = x.size(1)
    dtype = x.dtype
    if _HAS_CPP_MHC and hc == 4:
        return torch.ops.sgl_kernel.hc_head_fused_cpu(
            x.contiguous(),
            hc_fn.float().contiguous(),
            hc_scale.float().reshape(()),
            hc_base.float(),
            hc_eps,
            norm_eps,
        )
    shape = x.size()  # (T, hc, d)

    x_flat = x.flatten(1).float()  # [T, hc*d]
    rsqrt = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)  # [T, 1]
    mixes = F.linear(x_flat, hc_fn.float()) * rsqrt  # [T, hc]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps  # [T, hc]

    # y[t, k] = sum_j pre[t, j] * x[t, j, k]
    y = (pre.unsqueeze(-1) * x.float()).sum(dim=1)  # [T, hc, d] -> [T, d]
    return y.to(dtype)
