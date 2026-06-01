#!/bin/bash
# UV Environment Setup Script for Slime (Full Training Environment)
# One-to-one reproduction of build_conda.sh using uv venv instead of conda/mamba.
#
# This script installs:
#   - PyTorch 2.9.1 (CUDA 12.9)
#   - SGLang from source (specific commit) + router wheel
#   - Flash-Attention, TransformerEngine, Flash-Linear-Attention, Apex
#   - Megatron-LM from source (specific commit)
#   - mbridge, Megatron-Bridge, torch_memory_saver, nvidia-modelopt
#   - Slime + all runtime dependencies
#
# Options:
#   --mirror        Use Alibaba Cloud PyPI mirror for faster downloads

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
PYTHON_VERSION="3.12"

# UV paths on shared storage (CPFS) so that .venv symlinks work across cluster nodes
export UV_PYTHON_INSTALL_DIR="/mnt/cpfs/yangyicun/uv/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/cpfs/.cache/uv_cache}"
mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"

# Source commits (must match build_conda.sh)
SG_LANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

# Where to clone external source dependencies
DEPS_DIR="$(pwd)/.deps"
WHEEL_DIR="${DEPS_DIR}/wheels"
mkdir -p "${DEPS_DIR}" "${WHEEL_DIR}"

# ── Mirror support (off by default) ──────────────────────────
USE_MIRROR=false

# ── Patch tracking (only for git patches) ───────────────────
patch_marker() {
    echo "${DEPS_DIR}/${1}/.patched"
}

mark_done() {
    touch "$1"
}

is_done() {
    [ -f "$1" ]
}

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Helper: build index args for uv pip install (common packages)
uv_index_args() {
    if [ "$USE_MIRROR" = true ]; then
        echo "--index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.org/simple/"
    else
        echo ""
    fi
}

# Helper: ensure cuDNN headers/libs are visible to the C++ compiler
# System CUDA may not include cuDNN, but torch's dependency (nvidia-cudnn-cu12)
# installs it into the venv. We export its paths so that extensions like
# transformer_engine, flash-attn, and apex can find it.
ensure_cudnn_env() {
    local cudnn_include
    local cudnn_lib
    cudnn_include=$("$PYTHON" -c "
import nvidia.cudnn, pathlib
print(pathlib.Path(nvidia.cudnn.__path__[0]) / 'include')
" 2>/dev/null)
    cudnn_lib=$("$PYTHON" -c "
import nvidia.cudnn, pathlib
print(pathlib.Path(nvidia.cudnn.__path__[0]) / 'lib')
" 2>/dev/null)

    if [ -n "$cudnn_include" ] && [ -d "$cudnn_include" ] && [ -f "$cudnn_include/cudnn.h" ]; then
        export CUDNN_INCLUDE_DIR="$cudnn_include"
        export CUDNN_LIBRARY_DIR="$cudnn_lib"
        export CPLUS_INCLUDE_PATH="${CUDNN_INCLUDE_DIR}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
        export LIBRARY_PATH="${CUDNN_LIBRARY_DIR}${LIBRARY_PATH:+:$LIBRARY_PATH}"
        print_info "cuDNN found in venv: include=$cudnn_include, lib=$cudnn_lib"
    else
        print_warning "cuDNN pip package not found; relying on system CUDA for cudnn headers"
    fi
}

# ── Check if uv is installed ─────────────────────────────────
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv first."
        echo "Install options:"
        echo "  - curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  - pip install uv"
        exit 1
    fi
    print_success "uv is installed ($(uv --version))"
}

# ── Check Python version ─────────────────────────────────────
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python ${PYTHON_VERSION} or higher."
        exit 1
    fi
    local python_version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Found Python ${python_version}"
}

# ── Check system CUDA (required for building from source) ────
check_cuda() {
    if ! command -v nvcc &> /dev/null; then
        for cuda_dir in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12; do
            if [ -n "$cuda_dir" ] && [ -x "${cuda_dir}/bin/nvcc" ]; then
                export CUDA_HOME="$cuda_dir"
                export PATH="${CUDA_HOME}/bin:${PATH}"
                break
            fi
        done
    fi

    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc not found. Full training environment requires CUDA toolkit for building from source."
        exit 1
    fi
    print_success "nvcc found: $(nvcc --version | grep "release" | head -1)"

    if [[ -z "${CUDA_HOME:-}" ]]; then
        if [[ -d "/usr/local/cuda" ]]; then
            export CUDA_HOME="/usr/local/cuda"
            print_info "CUDA_HOME auto-set to $CUDA_HOME"
        else
            print_error "CUDA_HOME not set and /usr/local/cuda not found."
            exit 1
        fi
    else
        print_info "CUDA_HOME already set to $CUDA_HOME"
    fi
}

# ── Ensure uv-managed Python is installed on shared storage ──
ensure_uv_python() {
    if [[ "${VIRTUAL_ENV:-}" != "" ]]; then
        print_warning "Detected active virtual environment (${VIRTUAL_ENV}), deactivating..."
        deactivate 2>/dev/null || true
    fi
    print_info "Ensuring CPython ${PYTHON_VERSION} is installed in ${UV_PYTHON_INSTALL_DIR}..."
    uv python install "${PYTHON_VERSION}"
}

# ── Create virtual environment ───────────────────────────────
create_venv() {
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists at $VENV_DIR, skipping creation."
        return 0
    fi
    print_info "Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python "${PYTHON_VERSION}"
    print_success "Virtual environment created at $VENV_DIR"
}

ensure_pip() {
    if "$PYTHON" -m pip --version &>/dev/null; then
        return 0
    fi
    print_info "Installing pip into ${VENV_DIR} for packages that require native pip..."
    uv pip install --python "$PYTHON" $(uv_index_args) pip
}

# ── 1. Core PyTorch (CUDA 12.9) ─────────────────────────────
install_torch() {
    if "$PYTHON" -c "import torch" 2>/dev/null; then
        print_info "[1/16] torch already installed, skipping."
        return 0
    fi
    print_info "[1/16] Installing torch ecosystem (CUDA 12.9)..."
    if [ "$USE_MIRROR" = true ]; then
        uv pip install --python "$PYTHON" \
            torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
    else
        uv pip install --python "$PYTHON" \
            torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
    fi
    print_success "torch installed"
}

# ── 2. cuda-python (prevent sglang from pulling cuda 13.0) ──
install_cuda_python() {
    if "$PYTHON" -c "import cuda" 2>/dev/null; then
        print_info "[2/16] cuda-python already installed, skipping."
        return 0
    fi
    print_info "[2/16] Installing cuda-python==13.1.0 (blocks cuda 13.0)..."
    uv pip install --python "$PYTHON" $(uv_index_args) cuda-python==13.1.0
    print_success "cuda-python installed"
}

# ── 3. Build tools ───────────────────────────────────────────
install_build_tools() {
    if command -v cmake &>/dev/null && command -v ninja &>/dev/null; then
        print_info "[3/16] cmake + ninja already available, skipping."
        return 0
    fi
    print_info "[3/16] Installing cmake + ninja..."
    uv pip install --python "$PYTHON" $(uv_index_args) cmake ninja
    print_success "build tools installed"
}

# ── 4. SGLang from source ────────────────────────────────────
install_sglang() {
    if "$PYTHON" -c "import sglang" 2>/dev/null; then
        print_info "[4/16] SGLang already installed, skipping."
        return 0
    fi
    print_info "[4/16] Cloning & installing SGLang from source (commit ${SG_LANG_COMMIT})..."
    if [ ! -d "${DEPS_DIR}/sglang" ]; then
        git clone https://github.com/sgl-project/sglang.git "${DEPS_DIR}/sglang"
    fi
    cd "${DEPS_DIR}/sglang"
    git fetch origin
    git checkout "${SG_LANG_COMMIT}"
    uv pip install --python "$PYTHON" $(uv_index_args) -e "python[all]"
    cd - > /dev/null
    print_success "sglang installed from source"
}

# ── 5. Flash-Attention ───────────────────────────────────────
install_flash_attn() {
    if "$PYTHON" -c "import flash_attn" 2>/dev/null; then
        print_info "[5/16] Flash-Attention already installed, skipping."
        return 0
    fi
    local flash_wheel
    flash_wheel=$(find "${WHEEL_DIR}" -maxdepth 1 -type f -name "flash_attn-2.7.4.post1-*.whl" | sort | tail -1)
    if [ -n "$flash_wheel" ]; then
        print_info "[5/16] Installing Flash-Attention from cached wheel: ${flash_wheel}"
        uv pip install --python "$PYTHON" $(uv_index_args) "$flash_wheel"
        print_success "flash-attn installed from cached wheel"
        return 0
    fi
    print_info "[5/16] Installing Flash-Attention 2.7.4.post1 (this may take a while)..."
    print_warning "No cached flash-attn wheel found in ${WHEEL_DIR}; building from source and saving the wheel for next time."
    ensure_cudnn_env
    MAX_JOBS=64 uv pip install --python "$PYTHON" $(uv_index_args) --no-build-isolation flash-attn==2.7.4.post1
    local built_wheel
    built_wheel=$(find "${UV_CACHE_DIR}" -path "*/flash-attn/2.7.4.post1/*" -type f -name "flash_attn-2.7.4.post1-*.whl" 2>/dev/null | sort | tail -1)
    if [ -n "$built_wheel" ]; then
        cp -f "$built_wheel" "${WHEEL_DIR}/"
        sha256sum "${WHEEL_DIR}/$(basename "$built_wheel")" > "${WHEEL_DIR}/$(basename "$built_wheel").sha256"
        print_success "Saved flash-attn wheel for future installs: ${WHEEL_DIR}/$(basename "$built_wheel")"
    else
        print_warning "flash-attn installed, but no built wheel was found under ${UV_CACHE_DIR}"
    fi
    print_success "flash-attn installed"
}

# ── 6. mbridge ───────────────────────────────────────────────
install_mbridge() {
    if "$PYTHON" -c "import mbridge" 2>/dev/null; then
        print_info "[6/16] mbridge already installed, skipping."
        return 0
    fi
    print_info "[6/16] Installing mbridge..."
    uv pip install --python "$PYTHON" $(uv_index_args) --no-deps \
        git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c
    print_success "mbridge installed"
}

# ── 7. TransformerEngine ─────────────────────────────────────
install_transformer_engine() {
    if "$PYTHON" -c "import transformer_engine" 2>/dev/null; then
        print_info "[7/16] TransformerEngine already installed, skipping."
        return 0
    fi
    print_info "[7/16] Installing TransformerEngine 2.10.0..."
    ensure_cudnn_env
    uv pip install --python "$PYTHON" $(uv_index_args) --no-build-isolation \
        "transformer_engine[pytorch]==2.10.0"
    print_success "transformer_engine installed"
}

# ── 8. Flash-Linear-Attention ────────────────────────────────
install_flash_linear_attention() {
    if "$PYTHON" -c "import fla" 2>/dev/null; then
        print_info "[8/16] flash-linear-attention already installed, skipping."
        return 0
    fi
    print_info "[8/16] Installing flash-linear-attention 0.4.1..."
    uv pip install --python "$PYTHON" $(uv_index_args) flash-linear-attention==0.4.1
    print_success "flash-linear-attention installed"
}

# ── 9. Apex ──────────────────────────────────────────────────
install_apex() {
    if "$PYTHON" -c "import apex" 2>/dev/null; then
        print_info "[9/16] Apex already installed, skipping."
        return 0
    fi
    print_info "[9/16] Installing Apex from NVIDIA (this may take a while)..."
    ensure_cudnn_env
    ensure_pip
    # Apex uses --build-option which uv pip install does not support;
    # fall back to the venv's native pip for this step.
    local pip_index_args=""
    if [ "$USE_MIRROR" = true ]; then
        pip_index_args="--index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.org/simple/"
    fi
    local original_cuda_home
    local original_path
    local torch_cuda_version
    local nvcc_cuda_version
    original_cuda_home="${CUDA_HOME:-}"
    original_path="$PATH"
    torch_cuda_version=$("$PYTHON" -c "import torch; print(torch.version.cuda or '')")
    nvcc_cuda_version=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)
    if [ -n "$torch_cuda_version" ] && [ -n "$nvcc_cuda_version" ] && [ "$torch_cuda_version" != "$nvcc_cuda_version" ]; then
        local real_cuda_home
        local real_nvcc
        local apex_cuda_home
        real_cuda_home="$CUDA_HOME"
        real_nvcc=$(command -v nvcc)
        apex_cuda_home="${DEPS_DIR}/cuda-${torch_cuda_version}-for-apex"
        print_warning "Apex checks PyTorch CUDA ${torch_cuda_version} against nvcc ${nvcc_cuda_version}; using a temporary nvcc version wrapper for this build."
        rm -rf "$apex_cuda_home"
        mkdir -p "$apex_cuda_home/bin"
        for path in include lib lib64 nvvm; do
            if [ -e "${real_cuda_home}/${path}" ]; then
                ln -s "${real_cuda_home}/${path}" "${apex_cuda_home}/${path}"
            fi
        done
        cat > "${apex_cuda_home}/bin/nvcc" <<EOF
#!/bin/bash
if [[ "\$1" == "-V" || "\$1" == "--version" ]]; then
  cat <<'VERSION'
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release ${torch_cuda_version}, V${torch_cuda_version}.0
VERSION
  exit 0
fi
exec "${real_nvcc}" "\$@"
EOF
        chmod +x "${apex_cuda_home}/bin/nvcc"
        export CUDA_HOME="$apex_cuda_home"
        export PATH="${CUDA_HOME}/bin:${PATH}"
    fi
    set +e
    NVCC_APPEND_FLAGS="--threads 4" \
      "$PYTHON" -m pip install \
        $pip_index_args \
        --disable-pip-version-check --no-cache-dir \
        --no-build-isolation \
        --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
        git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
    local apex_status=$?
    set -e
    if [ -n "$original_cuda_home" ]; then
        export CUDA_HOME="$original_cuda_home"
    else
        unset CUDA_HOME
    fi
    export PATH="$original_path"
    if [ "$apex_status" -ne 0 ]; then
        return "$apex_status"
    fi
    print_success "apex installed"
}

# ── 10. torch_memory_saver ───────────────────────────────────
install_torch_memory_saver() {
    if "$PYTHON" -c "import torch_memory_saver" 2>/dev/null; then
        print_info "[10/16] torch_memory_saver already installed, skipping."
        return 0
    fi
    print_info "[10/16] Installing torch_memory_saver..."
    uv pip install --python "$PYTHON" $(uv_index_args) \
        --no-cache-dir --force-reinstall \
        git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b
    print_success "torch_memory_saver installed"
}

# ── 11. Megatron-Bridge ──────────────────────────────────────
install_megatron_bridge() {
    if "$PYTHON" -c "import megatron_bridge" 2>/dev/null; then
        print_info "[11/16] Megatron-Bridge already installed, skipping."
        return 0
    fi
    print_info "[11/16] Installing Megatron-Bridge (dev_rl)..."
    # nvidia-resiliency-ext (dependency of megatron-bridge) requires pybind11 at build time
    # when --no-build-isolation is used.
    uv pip install --python "$PYTHON" $(uv_index_args) pybind11
    uv pip install --python "$PYTHON" $(uv_index_args) --no-build-isolation \
        git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl
    print_success "Megatron-Bridge installed"
}

# ── 12. nvidia-modelopt ──────────────────────────────────────
install_nvidia_modelopt() {
    if "$PYTHON" -c "import modelopt" 2>/dev/null; then
        print_info "[12/16] nvidia-modelopt already installed, skipping."
        return 0
    fi
    print_info "[12/16] Installing nvidia-modelopt..."
    uv pip install --python "$PYTHON" $(uv_index_args) --no-build-isolation \
        "nvidia-modelopt[torch]>=0.37.0"
    print_success "nvidia-modelopt installed"
}

# ── 13. sglang-router (specific wheel) ───────────────────────
install_sglang_router() {
    if "$PYTHON" -c "from sglang.srt.utils import launch_router" 2>/dev/null || "$PYTHON" -c "import sglang_router" 2>/dev/null; then
        print_info "[13/16] sglang-router already installed, skipping."
        return 0
    fi
    print_info "[13/16] Installing sglang-router from wheel..."
    uv pip install --python "$PYTHON" $(uv_index_args) --force-reinstall \
        https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
    print_success "sglang-router installed"
}

# ── 14. Megatron-LM from source ──────────────────────────────
install_megatron_lm() {
    if "$PYTHON" -c "import megatron.core" 2>/dev/null; then
        print_info "[14/16] Megatron-LM already installed, skipping."
        return 0
    fi
    print_info "[14/16] Cloning & installing Megatron-LM from source (commit ${MEGATRON_COMMIT})..."
    if [ ! -d "${DEPS_DIR}/Megatron-LM" ]; then
        git clone --recursive https://github.com/NVIDIA/Megatron-LM.git "${DEPS_DIR}/Megatron-LM"
    fi
    cd "${DEPS_DIR}/Megatron-LM"
    git fetch origin
    git checkout "${MEGATRON_COMMIT}"
    # Ensure submodules are up-to-date
    git submodule update --init --recursive 2>/dev/null || true
    uv pip install --python "$PYTHON" $(uv_index_args) -e .
    cd - > /dev/null
    print_success "Megatron-LM installed from source"
}

# ── 15. Slime (editable + requirements) ──────────────────────
install_slime() {
    if "$PYTHON" -c "import slime, ray" 2>/dev/null; then
        print_info "[15/16] slime already installed, skipping."
        return 0
    fi
    print_info "[15/16] Installing slime in editable mode..."
    uv pip install --python "$PYTHON" $(uv_index_args) -e "."
    print_success "slime installed"
}

# ── 16. cudnn fix + numpy pin ────────────────────────────────
install_post_deps() {
    if "$PYTHON" -c "import importlib.metadata as md, numpy, nvidia.cudnn; assert numpy.__version__ == '1.26.4'; assert md.version('nvidia-cudnn-cu12') == '9.16.0.29'" 2>/dev/null; then
        print_info "[16/16] post-deps (numpy==1.26.4, nvidia-cudnn-cu12==9.16.0.29) already installed, skipping."
        return 0
    fi
    print_info "[16/16] Installing post-deps (nvidia-cudnn-cu12, numpy==1.26.4 for Megatron)..."
    uv pip install --python "$PYTHON" $(uv_index_args) nvidia-cudnn-cu12==9.16.0.29 "numpy==1.26.4"
    print_success "post-deps installed"
}

# ── Apply patches ────────────────────────────────────────────
apply_patches() {
    print_info "Applying patches to sglang and Megatron-LM..."
    local patch_dir
    patch_dir="$(pwd)/docker/patch/v0.5.9"

    local sglang_patch_marker
    sglang_patch_marker=$(patch_marker "sglang")
    if is_done "$sglang_patch_marker"; then
        print_info "sglang patch already applied, skipping."
    elif [ -d "${DEPS_DIR}/sglang" ] && [ -f "${patch_dir}/sglang.patch" ]; then
        cd "${DEPS_DIR}/sglang"
        git apply "${patch_dir}/sglang.patch" || print_warning "sglang patch may already be applied"
        cd - > /dev/null
        mark_done "$sglang_patch_marker"
        print_success "sglang patch applied"
    fi

    local megatron_patch_marker
    megatron_patch_marker=$(patch_marker "Megatron-LM")
    if is_done "$megatron_patch_marker"; then
        print_info "Megatron-LM patch already applied, skipping."
    elif [ -d "${DEPS_DIR}/Megatron-LM" ] && [ -f "${patch_dir}/megatron.patch" ]; then
        cd "${DEPS_DIR}/Megatron-LM"
        git apply "${patch_dir}/megatron.patch" || print_warning "megatron patch may already be applied"
        cd - > /dev/null
        mark_done "$megatron_patch_marker"
        print_success "Megatron-LM patch applied"
    fi
}

# ── Verification ─────────────────────────────────────────────
verify() {
    print_info "Verifying critical imports..."
    "$PYTHON" -c "import torch;         print(f'  torch:              {torch.__version__}  CUDA={torch.version.cuda}')"
    "$PYTHON" -c "import transformers;  print(f'  transformers:       {transformers.__version__}')"
    "$PYTHON" -c "import sglang;        print('  sglang:             OK')"
    "$PYTHON" -c "import flash_attn;    print('  flash_attn:         OK')"
    "$PYTHON" -c "import transformer_engine; print('  transformer_engine: OK')" 2>/dev/null || print_warning "  transformer_engine: import failed (may need manual check)"
    "$PYTHON" -c "import apex;          print('  apex:               OK')" 2>/dev/null || print_warning "  apex:               import failed (may need manual check)"
    "$PYTHON" -c "import datasets;      print(f'  datasets:           {datasets.__version__}')"
    "$PYTHON" -c "import wandb;         print(f'  wandb:              {wandb.__version__}')"
    "$PYTHON" -c "import importlib.metadata as md; assert md.version('nvidia-cudnn-cu12') == '9.16.0.29'; print(f'  nvidia-cudnn-cu12:  {md.version(\"nvidia-cudnn-cu12\")}')"
    "$PYTHON" -c "import numpy;         assert numpy.__version__ == '1.26.4'; print(f'  numpy:              {numpy.__version__}')"
    print_success "Verification complete"
}

# ── Usage ────────────────────────────────────────────────────
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mirror        Use Alibaba Cloud PyPI mirror for faster downloads"
    echo "  -c, --clean     Clean existing virtual environment and external deps before setup (keeps .deps/wheels)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Full training environment setup (official PyPI)"
    echo "  $0 --mirror     # Use Alibaba Cloud mirror"
    echo "  $0 -c           # Clean rebuild from scratch"
}

# ── Main ─────────────────────────────────────────────────────
main() {
    local clean=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --mirror)
                USE_MIRROR=true
                shift
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    if [ "$USE_MIRROR" = true ]; then
        print_info "Mirror mode ENABLED (Alibaba Cloud PyPI)"
    else
        print_info "Mirror mode DISABLED (using official PyPI)"
    fi

    print_info "Starting FULL training environment setup for slime (uv + build from source)..."
    print_info "Deps will be cloned into: ${DEPS_DIR}"

    check_uv
    check_python
    check_cuda
    ensure_uv_python

    if [ "$clean" = true ]; then
        if [ -d "$VENV_DIR" ]; then
            print_info "Cleaning virtual environment..."
            rm -rf "$VENV_DIR"
        fi
        if [ -d "$DEPS_DIR" ]; then
            print_info "Cleaning external dependencies (preserving cached wheels)..."
            find "$DEPS_DIR" -mindepth 1 -maxdepth 1 ! -name "$(basename "$WHEEL_DIR")" -exec rm -rf {} +
            mkdir -p "$DEPS_DIR" "$WHEEL_DIR"
        fi
    fi

    create_venv
    PYTHON="$(pwd)/$VENV_DIR/bin/python"

    install_torch
    install_cuda_python
    install_build_tools
    install_sglang
    install_flash_attn
    install_mbridge
    install_transformer_engine
    install_flash_linear_attention
    install_apex
    install_torch_memory_saver
    install_megatron_bridge
    install_nvidia_modelopt
    install_sglang_router
    install_megatron_lm
    install_slime
    install_post_deps
    apply_patches
    verify

    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo ""
    print_success "Full training environment setup complete! 🎉"
    echo ""
    print_success "Virtual environment is now ACTIVE!"
    echo ""
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
    echo "CUDA_HOME: ${CUDA_HOME}"
    echo ""
    echo "External source deps cloned in: ${DEPS_DIR}"
    echo "Cached wheels (flash-attn, etc.): ${WHEEL_DIR}"
    echo ""
    echo "To use uv run:"
    echo "  uv run python -m slime --help"
    echo ""
    echo "Or activate manually:"
    echo "  source $VENV_DIR/bin/activate"
}

main "$@"
