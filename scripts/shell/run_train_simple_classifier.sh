#!/bin/bash
set -e

# 简单图像分类器训练脚本（COCO格式标注，支持DDP、数据增强、LetterBox）
# 使用方式:
#   bash scripts/shell/run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml
#   bash scripts/shell/run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml --ngpus 2

CONFIG=""
NGPUS=1
OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    --ngpus) NGPUS="$2"; shift 2 ;;
    --override) OVERRIDE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    -h|--help)
      cat << 'EOF'
Usage: bash run_train_simple_classifier.sh --config <config.yaml> [OPTIONS]

Simple image classifier training script (COCO format, supports DDP, augmentation, LetterBox)

Required:
  --config <file>       Path to training config YAML file

Options:
  --ngpus N             Number of GPUs for DDP training (default: 1)
  --override "k=v,..."  Override config values (e.g., "epochs=100,batch_size=8")
  --master-port PORT    Distributed training master port (default: 29500)
  -h, --help            Show this help message

Examples:
  # Single GPU training
  bash run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml

  # Multi-GPU DDP training
  bash run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml --ngpus 2

  # Override config values
  bash run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml \
    --override "epochs=50,lr=0.001"

  # Specify master port for DDP
  bash run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml \
    --ngpus 4 --master-port 29501
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# 检查必需参数
if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required"
  cat << 'EOF'
Usage: bash scripts/shell/run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml \
[--ngpus 2] [--override "epochs=100,batch=8"] [--master-port 29500]
EOF
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: Config file not found: $CONFIG"
  exit 1
fi

echo "Starting simple classifier training..."
echo "  Config: ${CONFIG}"
echo "  GPUs: ${NGPUS}"
if [[ -n "$OVERRIDE" ]]; then
  echo "  Override: ${OVERRIDE}"
fi

OVERRIDE_ARG=""
if [[ -n "$OVERRIDE" ]]; then
  OVERRIDE_ARG="--override ${OVERRIDE}"
fi

if [ ! -d runs/simple_classifier ]; then
mkdir -p runs/simple_classifier;
fi

export PYTHONPATH=src:$PYTHONPATH
if [ "${NGPUS}" -gt 1 ]; then
  echo "Starting DDP training on ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" \
    --master_port="${MASTER_PORT:-29500}" \
    scripts/train/train_simple_classifier.py \
    --config "${CONFIG}" \
    ${OVERRIDE_ARG} 2>&1 | tee runs/simple_classifier/ddp_train.log
else
  echo "Starting single-GPU training..."
  python scripts/train/train_simple_classifier.py \
    --config "${CONFIG}" \
    ${OVERRIDE_ARG} 2>&1 | tee runs/simple_classifier/single_gpu_train.log
fi

echo "Training completed."
