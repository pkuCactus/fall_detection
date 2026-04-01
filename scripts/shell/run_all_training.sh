#!/bin/bash
set -e

# 完整的分阶段训练流水线
# 使用方式:
#   bash scripts/shell/run_all_training.sh
#   NGPUS=2 bash scripts/shell/run_all_training.sh

NGPUS="${NGPUS:-1}"

echo "=========================================="
echo "  Fall Detection Phased Training Pipeline"
echo "  GPUs: ${NGPUS}"
echo "=========================================="

echo ""
echo "==== Phase 1: Train Person Detector ===="
bash scripts/shell/run_train_detector.sh --ngpus "${NGPUS}"

echo ""
echo "==== Phase 2: Fine-tune Pose Estimator ===="
bash scripts/shell/run_train_pose.sh --ngpus "${NGPUS}"

echo ""
echo "==== Phase 3: Tune Tracker Parameters ===="
bash scripts/shell/run_tune_tracker.sh

echo ""
echo "==== Phase 4: Extract Classifier Features ===="
bash scripts/shell/run_extract_features.sh

echo ""
echo "==== Phase 5: Train Fall Classifier ===="
bash scripts/shell/run_train_classifier.sh --ngpus "${NGPUS}"

echo ""
echo "==== Phase 6: End-to-End Evaluation ===="
bash scripts/shell/run_evaluate_pipeline.sh

echo ""
echo "=========================================="
echo "  All training phases completed!"
echo "=========================================="
