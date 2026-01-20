FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]

ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG VER_SGLANG=main

RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    gcc \
    g++ \
    make \
    libsqlite3-dev \
    google-perftools \
    libtbb-dev \
    libnuma-dev \
    numactl

WORKDIR /opt

ENV UV_PYTHON_INSTALL_DIR=/usr/local/share/uv/python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/ && \
    uv venv --python 3.12

RUN echo -e '[[index]]\nname = "torch"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchvision"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "torchaudio"\nurl = "https://download.pytorch.org/whl/cpu"\n\n[[index]]\nname = "triton"\nurl = "https://download.pytorch.org/whl/cpu"' > .venv/uv.toml

ENV UV_CONFIG_FILE=/opt/.venv/uv.toml

WORKDIR /sgl-workspace

RUN curl -fsSL -o miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh && \
    bash miniforge.sh -b -p ./miniforge3 && \
    rm -f miniforge.sh && \
    . miniforge3/bin/activate && \
    conda install -y libsqlite==3.48.0 gperftools tbb libnuma numactl && \
    conda install --update-deps -c conda-forge -y gxx gcc sysroot_linux-64

ENV PATH=/sgl-workspace/miniforge3/bin:/sgl-workspace/miniforge3/condabin:${PATH}
ENV PIP_ROOT_USER_ACTION=ignore
ENV CONDA_PREFIX=/sgl-workspace/miniforge3

RUN pip config set global.index-url https://download.pytorch.org/whl/cpu && \
    pip config set global.extra-index-url https://pypi.org/simple

RUN git clone ${SGLANG_REPO} sglang && \
    cd sglang && \
    git checkout ${VER_SGLANG} && \
    cd python && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install . && \
    cd sglang/kernels/aot && \
    cp pyproject_cpu.toml pyproject.toml && \
    uv pip install . && \
    source /sgl-workspace/sglang/scripts/ci/utils/sgl_eval_ref.sh && \
    uv pip install "$SGL_EVAL_SPEC"

ENV SGLANG_USE_CPU_ENGINE=1
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:/usr/lib/x86_64-linux-gnu/libtbbmalloc.so:/opt/.venv/lib/libiomp5.so
ENV PATH="/opt/.venv/bin:$PATH"
RUN echo 'source /opt/.venv/bin/activate' >> /root/.bashrc

WORKDIR /sgl-workspace/sglang
