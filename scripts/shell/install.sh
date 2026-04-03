#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

VARIANT_ARGS=()
while [[ "$1" =~ ^-- ]]; do
    case "$1" in
        --variant)
            if [ -z "${2:-}" ]; then
                echo "Error: --variant requires a value (cpu, cu118, cu121, cu124, cu128)"
                exit 1
            fi
            VARIANT_ARGS=("--variant" "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Generating requirements.txt based on NVIDIA driver..."
python3 scripts/generate_requirements.py "${VARIANT_ARGS[@]}" -o requirements.txt

echo "Installing dependencies..."
pip install -r requirements.txt --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org

echo ""
echo "Installation complete. PyTorch version:"
python3 -c "import torch; print(' ', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
