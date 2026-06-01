#!/usr/bin/env bash
# UV environment setup script for NCFM.
#
# This script creates a uv-managed virtual environment and installs the
# dependencies required by the NCFM training, condensation, and evaluation
# scripts.
#
# Options:
#   --mirror                  Use Alibaba Cloud PyPI mirror for non-torch deps
#   --torch-backend BACKEND   PyTorch wheel backend: cu124, cu118, cpu, or default
#   --python VERSION          Python version managed by uv (default: 3.12)
#   -c, --clean               Remove the existing virtual environment first

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-cu124}"
USE_MIRROR=false
CLEAN=false

# Keep uv-managed Python and cache on shared storage, matching the cluster style
# used by setup_train_env_reference.sh.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/mnt/cpfs/yangyicun/uv/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/cpfs/.cache/uv_cache}"

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

show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --mirror                  Use Alibaba Cloud PyPI mirror for non-torch deps
  --torch-backend BACKEND   PyTorch backend: cu124, cu118, cpu, or default
  --python VERSION          Python version to install/use with uv (default: 3.12)
  -c, --clean               Remove existing virtual environment before setup
  -h, --help                Show this help message

Examples:
  $0
  $0 --mirror
  $0 --torch-backend cpu
  $0 --python 3.11 --clean
EOF
}

uv_index_args() {
    if [ "$USE_MIRROR" = true ]; then
        echo "--index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.org/simple/"
    else
        echo ""
    fi
}

torch_index_url() {
    case "$TORCH_BACKEND" in
        cu124)
            echo "https://download.pytorch.org/whl/cu124"
            ;;
        cu118)
            echo "https://download.pytorch.org/whl/cu118"
            ;;
        cpu)
            echo "https://download.pytorch.org/whl/cpu"
            ;;
        default)
            echo ""
            ;;
        *)
            print_error "Unknown --torch-backend: ${TORCH_BACKEND}"
            print_error "Expected one of: cu124, cu118, cpu, default"
            exit 1
            ;;
    esac
}

check_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        print_error "uv is not installed."
        echo "Install options:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  pip install uv"
        exit 1
    fi
    print_success "uv is installed ($(uv --version))"
}

check_project_files() {
    if [ ! -f "${PROJECT_ROOT}/requirements.txt" ]; then
        print_error "requirements.txt not found in ${PROJECT_ROOT}"
        exit 1
    fi
}

ensure_uv_dirs() {
    mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"
    print_info "UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}"
    print_info "UV_CACHE_DIR=${UV_CACHE_DIR}"
}

ensure_uv_python() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        print_warning "Detected active virtual environment (${VIRTUAL_ENV}); this script will use ${PROJECT_ROOT}/${VENV_DIR}."
    fi
    print_info "Ensuring CPython ${PYTHON_VERSION} is installed by uv..."
    uv python install "${PYTHON_VERSION}"
}

create_venv() {
    cd "${PROJECT_ROOT}"
    if [ "$CLEAN" = true ] && [ -d "$VENV_DIR" ]; then
        print_info "Removing existing virtual environment: ${VENV_DIR}"
        rm -rf "$VENV_DIR"
    fi

    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists at ${VENV_DIR}, skipping creation."
    else
        print_info "Creating virtual environment with uv..."
        uv venv "$VENV_DIR" --python "${PYTHON_VERSION}"
        print_success "Virtual environment created at ${VENV_DIR}"
    fi

    PYTHON="${PROJECT_ROOT}/${VENV_DIR}/bin/python"
    export PYTHON
}

install_torch() {
    local index_url
    index_url="$(torch_index_url)"

    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import torch
import torchvision
assert torch.__version__.split("+")[0] == "2.5.0"
assert torchvision.__version__.split("+")[0] == "0.20.0"
PY
    then
        print_info "PyTorch 2.5.0 and torchvision 0.20.0 already installed, skipping."
        return 0
    fi

    print_info "Installing PyTorch 2.5.0 / torchvision 0.20.0 (backend: ${TORCH_BACKEND})..."
    if [ -n "$index_url" ]; then
        uv pip install --python "$PYTHON" \
            --index-url "$index_url" \
            torch==2.5.0 torchvision==0.20.0
    else
        uv pip install --python "$PYTHON" \
            torch==2.5.0 torchvision==0.20.0
    fi
    print_success "PyTorch installed"
}

install_requirements() {
    print_info "Installing project dependencies from requirements.txt..."
    uv pip install --python "$PYTHON" $(uv_index_args) -r "${PROJECT_ROOT}/requirements.txt"
    print_success "Project dependencies installed"
}

verify() {
    print_info "Verifying critical imports..."
    "$PYTHON" - <<'PY'
import importlib.metadata as md

import efficientnet_pytorch
import matplotlib
import numpy
import torch
import torchvision
import tqdm
import yaml

print(f"  python:      {tuple(__import__('sys').version_info[:3])}")
print(f"  torch:       {torch.__version__}  CUDA={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu:         {torch.cuda.get_device_name(0)}")
print(f"  torchvision: {torchvision.__version__}")
print(f"  numpy:       {numpy.__version__}")
print(f"  matplotlib:  {matplotlib.__version__}")
print(f"  PyYAML:      {md.version('PyYAML')}")
print(f"  tqdm:        {tqdm.__version__}")
print("  efficientnet_pytorch: OK")
PY
    print_success "Verification complete"
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mirror)
                USE_MIRROR=true
                shift
                ;;
            --torch-backend)
                TORCH_BACKEND="${2:-}"
                shift 2
                ;;
            --python)
                PYTHON_VERSION="${2:-}"
                shift 2
                ;;
            -c|--clean)
                CLEAN=true
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

    if [ -z "$PYTHON_VERSION" ]; then
        print_error "--python requires a version value"
        exit 1
    fi

    print_info "Starting NCFM environment setup with uv..."
    print_info "Project root: ${PROJECT_ROOT}"
    print_info "Python version: ${PYTHON_VERSION}"
    print_info "Torch backend: ${TORCH_BACKEND}"

    check_uv
    check_project_files
    ensure_uv_dirs
    ensure_uv_python
    create_venv
    install_torch
    install_requirements
    verify

    echo ""
    print_success "NCFM environment setup complete."
    echo "Activate it with:"
    echo "  source ${VENV_DIR}/bin/activate"
    echo ""
    echo "Example:"
    echo "  cd condense"
    echo "  torchrun --nproc_per_node=1 --nnodes=1 condense_script.py --gpu=\"0\" --ipc=1 --config_path=../config/ipc1/cifar10.yaml"
}

main "$@"
