#!/bin/bash
set -e

# 测试验证脚本
# 使用方式:
#   bash scripts/shell/run_tests.sh
#   bash scripts/shell/run_tests.sh tests/test_training_scripts.py

TEST_TARGET="${1:-tests/}"

echo "Running tests: ${TEST_TARGET}"
PYTHONPATH=src pytest "${TEST_TARGET}" -v
