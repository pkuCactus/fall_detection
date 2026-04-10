#!/bin/bash
set -e

# 完整的7阶段训练流水线
# 使用方式:
#   bash scripts/shell/run_all_training.sh
#   NGPUS=2 bash scripts/shell/run_all_training.sh
#   bash scripts/shell/run_all_training.sh --skip-phase 3,4
#   bash scripts/shell/run_all_training.sh --detector-config configs/training/detector.yaml

# 默认参数
NGPUS="${NGPUS:-1}"
SKIP_PHASES=""
DETECTOR_CONFIG="configs/training/detector.yaml"
POSE_CONFIG="configs/training/pose.yaml"
CLASSIFIER_CONFIG="configs/training/classifier.yaml"
SIMPLE_CLASSIFIER_CONFIG="configs/training/simple_classifier.yaml"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --ngpus) NGPUS="$2"; shift 2 ;;
    --skip-phase) SKIP_PHASES="$2"; shift 2 ;;
    --detector-config) DETECTOR_CONFIG="$2"; shift 2 ;;
    --pose-config) POSE_CONFIG="$2"; shift 2 ;;
    --classifier-config) CLASSIFIER_CONFIG="$2"; shift 2 ;;
    --simple-classifier-config) SIMPLE_CLASSIFIER_CONFIG="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# 辅助函数：检查是否跳过某个阶段
should_skip() {
  local phase=$1
  if [[ "$SKIP_PHASES" == *"$phase"* ]]; then
    return 0  # true, should skip
  else
    return 1  # false, should run
  fi
}

# 辅助函数：检查配置文件是否存在
check_config() {
  local config=$1
  if [[ ! -f "$config" ]]; then
    echo "Error: Config file not found: $config"
    exit 1
  fi
}

# 辅助函数：检查脚本是否存在
check_script() {
  local script=$1
  if [[ ! -f "$script" ]]; then
    echo "Error: Script not found: $script"
    exit 1
  fi
}

# 开始时间
START_TIME=$(date +%s)

echo "=========================================="
echo "  Fall Detection 7-Phase Training Pipeline"
echo "  Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPUs: ${NGPUS}"
echo "  Skip Phases: ${SKIP_PHASES:-none}"
echo "=========================================="

# 阶段1：人体检测器训练
if should_skip "1"; then
  echo ""
  echo "==== Phase 1: SKIPPED ===="
else
  echo ""
  echo "==== Phase 1: Train Person Detector ===="
  echo "Config: ${DETECTOR_CONFIG}"
  check_config "${DETECTOR_CONFIG}"
  check_script "scripts/shell/run_train_detector.sh"
  
  PHASE_START=$(date +%s)
  bash scripts/shell/run_train_detector.sh --config "${DETECTOR_CONFIG}"
  PHASE_END=$(date +%s)
  
  echo "Phase 1 completed in $((PHASE_END - PHASE_START)) seconds"
  
  # 检查输出
  if [[ -f "outputs/detector/best.pt" ]]; then
    echo "✓ Detector weights saved to outputs/detector/best.pt"
  else
    echo "⚠ Warning: Detector weights not found at outputs/detector/best.pt"
  fi
fi

# 阶段2：姿态估计器微调
if should_skip "2"; then
  echo ""
  echo "==== Phase 2: SKIPPED ===="
else
  echo ""
  echo "==== Phase 2: Fine-tune Pose Estimator ===="
  echo "Config: ${POSE_CONFIG}"
  check_config "${POSE_CONFIG}"
  check_script "scripts/shell/run_train_pose.sh"
  
  PHASE_START=$(date +%s)
  bash scripts/shell/run_train_pose.sh --config "${POSE_CONFIG}"
  PHASE_END=$(date +%s)
  
  echo "Phase 2 completed in $((PHASE_END - PHASE_START)) seconds"
  
  # 检查输出
  if [[ -f "outputs/pose/best.pt" ]]; then
    echo "✓ Pose estimator weights saved to outputs/pose/best.pt"
  else
    echo "⚠ Warning: Pose weights not found at outputs/pose/best.pt"
  fi
fi

# 阶段3：跟踪器参数调优
if should_skip "3"; then
  echo ""
  echo "==== Phase 3: SKIPPED ===="
else
  echo ""
  echo "==== Phase 3: Tune Tracker Parameters ===="
  echo "Note: This phase requires video data in data/videos/"
  check_script "scripts/shell/run_tune_tracker.sh"
  
  PHASE_START=$(date +%s)
  bash scripts/shell/run_tune_tracker.sh
  PHASE_END=$(date +%s)
  
  echo "Phase 3 completed in $((PHASE_END - PHASE_START)) seconds"
  
  # 检查输出
  if [[ -f "outputs/tracker/tune_result.json" ]]; then
    echo "✓ Tracker tuning results saved to outputs/tracker/tune_result.json"
  else
    echo "⚠ Warning: Tracker tuning results not found"
  fi
fi

# 阶段4：特征提取（分类器训练数据准备）
if should_skip "4"; then
  echo ""
  echo "==== Phase 4: SKIPPED ===="
else
  echo ""
  echo "==== Phase 4: Extract Classifier Features ===="
  echo "Note: This phase requires detector and pose models from phases 1-2"
  echo "      and video data with labels in data/labels.json"
  check_script "scripts/shell/run_extract_features.sh"
  
  PHASE_START=$(date +%s)
  bash scripts/shell/run_extract_features.sh
  PHASE_END=$(date +%s)
  
  echo "Phase 4 completed in $((PHASE_END - PHASE_START)) seconds"
  
  # 检查输出
  if [[ -d "outputs/cache" ]] && [[ $(ls -A outputs/cache/*.npz 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "✓ Feature cache files saved to outputs/cache/"
    echo "  Total files: $(ls outputs/cache/*.npz 2>/dev/null | wc -l)"
  else
    echo "⚠ Warning: No feature cache files found in outputs/cache/"
  fi
fi

# 阶段5：融合分类器训练
if should_skip "5"; then
  echo ""
  echo "==== Phase 5: SKIPPED ===="
else
  echo ""
  echo "==== Phase 5: Train Fusion Classifier ===="
  echo "Config: ${CLASSIFIER_CONFIG}"
  echo "GPUs: ${NGPUS}"
  check_config "${CLASSIFIER_CONFIG}"
  check_script "scripts/shell/run_train_classifier.sh"
  
  # 检查特征缓存
  if [[ ! -d "outputs/cache" ]] || [[ $(ls -A outputs/cache/*.npz 2>/dev/null | wc -l) -eq 0 ]]; then
    echo "⚠ Warning: No feature cache found. Run Phase 4 first or skip Phase 5."
  fi
  
  PHASE_START=$(date +%s)
  bash scripts/shell/run_train_classifier.sh --ngpus "${NGPUS}"
  PHASE_END=$(date +%s)
  
  echo "Phase 5 completed in $((PHASE_END - PHASE_START)) seconds"
  
  # 检查输出
  if [[ -f "outputs/classifier/best.pt" ]]; then
    echo "✓ Fusion classifier weights saved to outputs/classifier/best.pt"
  else
    echo "⚠ Warning: Fusion classifier weights not found at outputs/classifier/best.pt"
  fi
fi

# 阶段6：简单分类器训练
if should_skip "6"; then
  echo ""
  echo "==== Phase 6: SKIPPED ===="
else
  echo ""
  echo "==== Phase 6: Train Simple Classifier ===="
  echo "Config: ${SIMPLE_CLASSIFIER_CONFIG}"
  echo "GPUs: ${NGPUS}"
  check_config "${SIMPLE_CLASSIFIER_CONFIG}"
  check_script "scripts/shell/run_train_simple_classifier.sh"
  
  PHASE_START=$(date +%s)
  bash scripts/shell/run_train_simple_classifier.sh \
    --config "${SIMPLE_CLASSIFIER_CONFIG}" \
    --ngpus "${NGPUS}"
  PHASE_END=$(date +%s)
  
  echo "Phase 6 completed in $((PHASE_END - PHASE_START)) seconds"
  
  # 检查输出
  if [[ -f "outputs/simple_classifier/best.pt" ]]; then
    echo "✓ Simple classifier weights saved to outputs/simple_classifier/best.pt"
  else
    echo "⚠ Warning: Simple classifier weights not found at outputs/simple_classifier/best.pt"
  fi
fi

# 阶段7：端到端评估与阈值搜索
if should_skip "7"; then
  echo ""
  echo "==== Phase 7: SKIPPED ===="
else
  echo ""
  echo "==== Phase 7: End-to-End Evaluation ===="
  echo "Note: This phase requires:"
  echo "  - Detector weights (Phase 1)"
  echo "  - Pose weights (Phase 2)"
  echo "  - Classifier weights (Phase 5 or 6)"
  echo "  - Video data in data/videos/"
  echo "  - Ground truth labels in data/event_gt.json"
  check_script "scripts/shell/run_evaluate_pipeline.sh"
  
  PHASE_START=$(date +%s)
  bash scripts/shell/run_evaluate_pipeline.sh
  PHASE_END=$(date +%s)
  
  echo "Phase 7 completed in $((PHASE_END - PHASE_START)) seconds"
  
  # 检查输出
  if [[ -f "outputs/eval/eval_result.json" ]]; then
    echo "✓ Evaluation results saved to outputs/eval/eval_result.json"
  else
    echo "⚠ Warning: Evaluation results not found at outputs/eval/eval_result.json"
  fi
fi

# 结束时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "  All training phases completed!"
echo "  Total Time: $((TOTAL_TIME / 3600))h $(((TOTAL_TIME % 3600) / 60))m $((TOTAL_TIME % 60))s"
echo "  End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 输出训练产物摘要
echo ""
echo "Training Artifacts Summary:"
echo "  - Detector: outputs/detector/best.pt"
echo "  - Pose Estimator: outputs/pose/best.pt"
echo "  - Fusion Classifier: outputs/classifier/best.pt"
echo "  - Simple Classifier: outputs/simple_classifier/best.pt"
echo "  - Tracker Tuning: outputs/tracker/tune_result.json"
echo "  - Feature Cache: outputs/cache/*.npz"
echo "  - Evaluation: outputs/eval/eval_result.json"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results in outputs/eval/eval_result.json"
echo "  2. Adjust thresholds in configs/pipeline/default.yaml based on evaluation"
echo "  3. Export models for deployment using scripts/tools/export_yoloworld.py"
echo "  4. Run inference demo using scripts/demo/run_pipeline_demo.py"