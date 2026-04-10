#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Parse arguments
VARIANT=""
DEV=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            if [ -z "${2:-}" ]; then
                echo "Error: --variant requires a value (cpu, cu118, cu121, cu124, cu128)"
                exit 1
            fi
            VARIANT="$2"
            shift 2
            ;;
        --no-dev)
            DEV=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect CUDA variant from NVIDIA driver if not specified
if [ -z "$VARIANT" ]; then
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)
        echo "Detected NVIDIA driver major version: $DRIVER_VERSION"
        
        if [ "$DRIVER_VERSION" -ge 525 ]; then
            VARIANT="cu124"
        elif [ "$DRIVER_VERSION" -ge 520 ]; then
            VARIANT="cu121"
        elif [ "$DRIVER_VERSION" -ge 450 ]; then
            VARIANT="cu118"
        else
            VARIANT="cpu"
            echo "Warning: Driver version too old, using CPU variant"
        fi
        echo "Auto-selected variant: $VARIANT"
    else
        VARIANT="cpu"
        echo "No NVIDIA driver detected, using CPU variant"
    fi
fi

# Build install command
EXTRA_INDEX_URL=""
if [[ "$VARIANT" != "cpu" ]]; then
    EXTRA_INDEX_URL="--extra-index-url https://download.pytorch.org/whl/${VARIANT}"
fi

INSTALL_OPTS="-e ."
if [ "$DEV" = true ]; then
    INSTALL_OPTS="$INSTALL_OPTS,[dev]"
fi
INSTALL_OPTS="$INSTALL_OPTS,[torch-${VARIANT}]"

echo ""
echo "=========================================="
echo "Installing fall_detection"
echo "  Variant: $VARIANT"
echo "  Dev tools: $DEV"
echo "=========================================="
echo ""

# Install using pyproject.toml
# The --extra-index-url ensures we get the correct CUDA version of PyTorch
if [ -n "$EXTRA_INDEX_URL" ]; then
    echo "Installing PyTorch with CUDA $VARIANT from PyTorch wheel index..."
    pip install $EXTRA_INDEX_URL "$INSTALL_OPTS" \
        --trusted-host download.pytorch.org \
        --trusted-host download-r2.pytorch.org
else
    echo "Installing PyTorch CPU version..."
    pip install "$INSTALL_OPTS"
fi

# Install CLIP separately (not on PyPI)
echo ""
echo "Installing CLIP from GitHub..."
pip install git+https://github.com/openai/CLIP.git

echo ""
echo "Installation complete!"
echo ""
echo "PyTorch version:"
python3 -c "import torch; print('  Version:', torch.__version__)"
python3 -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -c "import torch; print('  CUDA version:', torch.version.cuda)"
fi

echo ""
echo "Usage:"
echo "  # Recommended install (auto-detect CUDA):"
echo "  bash scripts/shell/install.sh"
echo ""
echo "  # Specific CUDA variant:"
echo "  bash scripts/shell/install.sh --variant cu124"
echo ""
echo "  # CPU only:"
echo "  bash scripts/shell/install.sh --variant cpu"
echo ""
echo "  # Without dev tools:"
echo "  bash scripts/shell/install.sh --no-dev"