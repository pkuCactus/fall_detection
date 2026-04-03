#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Generating requirements.txt based on NVIDIA driver..."
python3 scripts/generate_requirements.py -o requirements.txt

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Installation complete. PyTorch version:"
python3 -c "import torch; print(' ', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
