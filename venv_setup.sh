#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_PYTHON="3.10"
DEFAULT_VENV_DIR="${ROOT_DIR}/.venv-geqtrain"

PYTHON_VERSION="${DEFAULT_PYTHON}"
VENV_DIR="${DEFAULT_VENV_DIR}"
TORCH_BACKEND="auto"
TORCH_VERSION=""
INSTALL_TORCH=1
INSTALL_DEV=0
RECREATE_VENV=0

usage() {
  cat <<USAGE
Usage: ./venv_setup.sh [options]

Create a uv-based virtual environment for GEqTrain and install this repository
in editable mode.

Options:
  --python VERSION         Python version to use (default: ${DEFAULT_PYTHON})
  --venv-dir PATH          Virtual environment directory (default: ${DEFAULT_VENV_DIR})
  --torch-backend BACKEND  uv torch backend: auto|cpu|cu118|cu121|cu124|cu126|cu128|rocm (default: auto)
  --torch-version VERSION  Torch version to install (default: latest)
  --no-torch               Skip torch installation
  --dev                    Install developer/test dependencies
  --recreate               Remove existing venv before creating a new one
  -h, --help               Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --torch-backend)
      TORCH_BACKEND="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --no-torch)
      INSTALL_TORCH=0
      shift
      ;;
    --dev)
      INSTALL_DEV=1
      shift
      ;;
    --recreate)
      RECREATE_VENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -d "${HOME}/.local/bin" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  if [[ -d "${HOME}/.cargo/bin" ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to find uv after installation. Add ~/.local/bin or ~/.cargo/bin to PATH and retry." >&2
    exit 1
  fi
}

ensure_uv

if [[ "${RECREATE_VENV}" -eq 1 && -d "${VENV_DIR}" ]]; then
  echo "Removing existing venv: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR} (Python ${PYTHON_VERSION})"
  if ! uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed; then
    echo "Python ${PYTHON_VERSION} not available locally. Installing via uv..."
    uv python install "${PYTHON_VERSION}"
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed
  fi
else
  echo "Using existing venv: ${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python binary not found in venv: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Installing base packaging tools..."
uv pip install --python "${PYTHON_BIN}" -U pip setuptools wheel

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  TORCH_SPEC="torch"
  if [[ -n "${TORCH_VERSION}" ]]; then
    TORCH_SPEC="torch==${TORCH_VERSION}"
  fi

  echo "Installing ${TORCH_SPEC} (backend: ${TORCH_BACKEND})"
  if [[ "${TORCH_BACKEND}" == "auto" ]]; then
    uv pip install --python "${PYTHON_BIN}" "${TORCH_SPEC}"
  else
    uv pip install --python "${PYTHON_BIN}" "${TORCH_SPEC}" --torch-backend "${TORCH_BACKEND}"
  fi
fi

echo "Installing GEqTrain (editable)..."
uv pip install --python "${PYTHON_BIN}" -e "${ROOT_DIR}"

echo "Installing runtime and tutorial dependencies..."
RUNTIME_DEPS=(
  scipy
  matplotlib
  pandas
  plotly
)
uv pip install --python "${PYTHON_BIN}" "${RUNTIME_DEPS[@]}"

if [[ "${INSTALL_DEV}" -eq 1 ]]; then
  echo "Installing developer dependencies..."
  uv pip install --python "${PYTHON_BIN}" pytest ruff
fi

echo
echo "Setup complete."
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Quick checks:"
echo "  geqtrain-train --help"
echo "  geqtrain-test-equivariance --help"
