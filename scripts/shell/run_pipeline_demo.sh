#!/bin/bash

# Pipeline演示脚本
# 展示完整的跌倒检测Pipeline
# 支持单视频文件或目录（自动处理目录下所有视频）
# 根据 --config 参数自动区分标准Pipeline或YOLO-World Pipeline

INPUT="${1:-data/sample.mp4}"
OUTPUT="${2:-}"
shift 2 || true

DEMO_SCRIPT="scripts/demo/run_pipeline_demo.py"
DEFAULT_CONFIG="configs/pipeline/default.yaml"

# 如果用户没有显式指定 --config，补充默认配置
has_config=false
for arg in "$@"; do
    if [ "$arg" = "--config" ]; then
        has_config=true
        break
    fi
done
if [ "$has_config" = false ]; then
    set -- "$@" --config "$DEFAULT_CONFIG"
fi

# 支持的视频格式
VIDEO_EXTENSIONS="mp4|avi|mov|mkv|flv|wmv|webm|m4v|3gp"

# 检查输入是否为目录
if [ -d "$INPUT" ]; then
    echo "=========================================="
    echo "Fall Detection Pipeline - Batch Processing"
    echo "Input directory: $INPUT"
    echo "=========================================="
    echo ""

    # 查找所有视频文件
    VIDEO_COUNT=0
    while IFS= read -r video_file; do
        VIDEO_COUNT=$((VIDEO_COUNT + 1))
    done < <(find "$INPUT" -type f -regextype posix-extended -iregex ".*\.($VIDEO_EXTENSIONS)$" 2>/dev/null | sort)

    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo "Error: No video files found in directory: $INPUT"
        echo "Supported formats: $VIDEO_EXTENSIONS"
        exit 1
    fi

    echo "Found $VIDEO_COUNT video(s) to process"
    echo ""

    # 确定输出目录
    if [ -z "$OUTPUT" ]; then
        OUTPUT_DIR="outputs/demo_batch"
    else
        OUTPUT_DIR="$OUTPUT"
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    # 处理每个视频
    PROCESSED=0
    FAILED=0

    while IFS= read -r video_file; do
        PROCESSED=$((PROCESSED + 1))

        # 计算相对于输入目录的相对路径，保持子目录结构
        rel_path=$(realpath --relative-to="$INPUT" "$video_file")
        rel_dir=$(dirname "$rel_path")
        video_basename=$(basename "$video_file")
        video_name="${video_basename%.*}"

        # 生成输出文件名（保留原始子目录结构）
        output_file="$OUTPUT_DIR/$rel_dir/${video_name}_output.mp4"
        mkdir -p "$(dirname "$output_file")"

        echo "----------------------------------------"
        echo "[$PROCESSED/$VIDEO_COUNT] Processing: $rel_path"
        echo "  Input:  $video_file"
        echo "  Output: $output_file"
        echo ""

        # 运行pipeline
        if python "$DEMO_SCRIPT" --video "$video_file" --output "$output_file" "$@" --headless; then
            echo "  ✓ Completed: $video_basename"
        else
            echo "  ✗ Failed: $video_basename"
            FAILED=$((FAILED + 1))
        fi
        echo ""

    done < <(find "$INPUT" -type f -regextype posix-extended -iregex ".*\.($VIDEO_EXTENSIONS)$" 2>/dev/null | sort)

    echo "=========================================="
    echo "Batch Processing Complete"
    echo "=========================================="
    echo "Total videos: $VIDEO_COUNT"
    echo "Successful:   $((VIDEO_COUNT - FAILED))"
    echo "Failed:       $FAILED"
    echo "Output directory: $OUTPUT_DIR"

    if [ "$FAILED" -gt 0 ]; then
        exit 1
    fi

else
    # 单文件处理模式
    VIDEO="$INPUT"

    # 检查文件是否存在
    if [ ! -f "$VIDEO" ]; then
        echo "Error: Video file not found: $VIDEO"
        exit 1
    fi

    # 确定输出文件
    if [ -z "$OUTPUT" ]; then
        # 自动生成输出文件名
        video_dir=$(dirname "$VIDEO")
        video_basename=$(basename "$VIDEO")
        video_name="${video_basename%.*}"
        OUTPUT="$video_dir/${video_name}_output.mp4"
    else
        OUTPUT="$OUTPUT/$(basename "${VIDEO%.*}_output.mp4")"
    fi

    echo "=========================================="
    echo "Fall Detection Pipeline Demo"
    echo "=========================================="
    echo "Video:  $VIDEO"
    echo "Output: $OUTPUT"
    echo ""
    echo "Controls:"
    echo "  ESC - quit"
    echo "  p   - pause/resume"
    echo "  s   - save current frame"
    echo ""

    python "$DEMO_SCRIPT" --video "$VIDEO" --output "$OUTPUT" "$@"
fi
