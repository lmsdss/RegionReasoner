#!/bin/bash
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [TEST_DATA_PATH] [MODEL_PATH] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  TEST_DATA_PATH    Path to test data file (optional)"
    echo "  MODEL_PATH        Path to model directory (optional)"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 refcocog.json /path/to/custom/model"
    echo "  $0 refcocog.json pretrained"
    echo "  $0 refcocog.json trained"
    echo ""
    echo "Available test files:"
    echo "  - refcocoplus_merged.json"
    echo "  - refcocog_merged.json"
    echo ""
    echo "Model path shortcuts:"
    echo "  - 'pretrained': Use pretrained VisionReasoner-7B model"
    echo "  - 'trained': Use fine-tuned model from training"
    exit 1
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

MODEL_TYPE="vision_reasoner"  # Model type: qwen or vision_reasoner or qwen2

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided, use defaults
    TEST_DATA_PATH="/sunwenfang/VisionReasoner/evaluation/refcocog_merged.json"
    MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/splitfull_reward2_1_0d5/global_step_658/actor/huggingface"
    echo "No arguments provided, using defaults:"
    echo "  - Test data: refcocog_merged.json"
    echo "  - Model: trained model"
elif [ $# -eq 1 ]; then
    # One argument provided (test data path)
    if [[ "$1" == /* ]]; then
        # Absolute path provided
        TEST_DATA_PATH="$1"
    else
        # Relative filename provided, construct full path
        TEST_DATA_PATH="/sunwenfang/VisionReasoner/evaluation/$1"
    fi
    # Use default model path
    MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/splitfull_reward2_1_0d5/global_step_658/actor/huggingface"
    echo "Using test data: $(basename $TEST_DATA_PATH)"
    echo "Using default model: trained model"
elif [ $# -eq 2 ]; then
    # Two arguments provided (test data path and model path)
    if [[ "$1" == /* ]]; then
        TEST_DATA_PATH="$1"
    else
        TEST_DATA_PATH="/sunwenfang/VisionReasoner/evaluation/$1"
    fi
    
    # Parse model path argument
    case "$2" in
        "pretrained")
            MODEL_PATH="/sunwenfang/VisionReasoner/pretrained_models/VisionReasoner-7B"
            echo "Using pretrained VisionReasoner-7B model"
            ;;
        "trained")
            MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/splitfull_reward2_1_0d5/global_step_658/actor/huggingface"
            echo "Using fine-tuned model"
            ;;
        /*)
            # Absolute path provided
            MODEL_PATH="$2"
            echo "Using custom model path: $MODEL_PATH"
            ;;
        *)
            # Assume it's a relative path or shortcut we don't recognize
            echo "Warning: Unrecognized model shortcut '$2', treating as relative path"
            MODEL_PATH="/sunwenfang/VisionReasoner/$2"
            ;;
    esac
    
    echo "Using test data: $(basename $TEST_DATA_PATH)"
else
    echo "Error: Too many arguments provided (maximum 2)"
    usage
fi

# Validate that the test data file exists
if [ ! -f "$TEST_DATA_PATH" ]; then
    echo "Error: Test data file not found: $TEST_DATA_PATH"
    echo ""
    echo "Available files in evaluation directory:"
    ls -la /sunwenfang/VisionReasoner/evaluation/*.json 2>/dev/null | grep -E '\.(json)$' || echo "No JSON files found"
    exit 1
fi

# Validate that the model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo ""
    echo "Please check if the model directory exists or use one of the shortcuts:"
    echo "  - 'pretrained' for VisionReasoner-7B"
    echo "  - 'trained' for fine-tuned model"
    exit 1
fi

# Extract filename for output directory naming
TEST_FILENAME=$(basename "$TEST_DATA_PATH" .json)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_PATH="/sunwenfang/VisionReasoner/detection_eval_results/test_multi/${TEST_FILENAME}_${MODEL_NAME}_$(date +%m%d%H%M)"

# Create log directory and file
LOG_DIR="$OUTPUT_PATH/logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/eval_log_${TIMESTAMP}_${TEST_FILENAME}.log"

# Log script start
echo "=================================" | tee -a $LOG_FILE
echo "Evaluation Started: $(date)" | tee -a $LOG_FILE
echo "Model Type: $MODEL_TYPE" | tee -a $LOG_FILE
echo "Model Path: $MODEL_PATH" | tee -a $LOG_FILE
echo "Test Data Path: $TEST_DATA_PATH" | tee -a $LOG_FILE
echo "Output Path: $OUTPUT_PATH" | tee -a $LOG_FILE
echo "Log File: $LOG_FILE" | tee -a $LOG_FILE
echo "=================================" | tee -a $LOG_FILE

# Customize GPU array here - specify which GPUs to use
# GPU_ARRAY=(0 1 2 3 4 5 6 7)  # Example: using GPUs 0, 1, 2, 3
GPU_ARRAY=(0 1 2 3)  # Example: using GPUs 0, 1, 2, 3
# GPU_ARRAY=(2 3 6 7)  # Example: using GPUs 0, 1, 2, 3

NUM_PARTS=${#GPU_ARRAY[@]}

echo "Using GPUs: ${GPU_ARRAY[*]}" | tee -a $LOG_FILE
echo "Number of parallel processes: $NUM_PARTS" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Create output directory
mkdir -p $OUTPUT_PATH

export HF_ENDPOINT=https://hf-mirror.com

# Run processes in parallel
echo "Starting parallel evaluation processes..." | tee -a $LOG_FILE
for i in $(seq 0 $((NUM_PARTS-1))); do
    gpu_id=${GPU_ARRAY[$i]}
    process_idx=$i  # 0-based indexing for process
    
    echo "Starting process $process_idx on GPU $gpu_id" | tee -a $LOG_FILE
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    (
        echo "Process $process_idx: Evaluation started at $(date)" >> $LOG_FILE
        python evaluation/evaluation_multi_segmentation.py \
            --model $MODEL_TYPE \
            --model_path $MODEL_PATH \
            --test_data_path $TEST_DATA_PATH \
            --vis_output_path $OUTPUT_PATH/${TEST_FILENAME}_visualizations \
            --output_path $OUTPUT_PATH \
            --idx $process_idx \
            --num_parts $NUM_PARTS \
            --batch_size 2 2>&1 | tee -a "$LOG_DIR/process_${process_idx}_gpu_${gpu_id}.log"
        echo "Process $process_idx: Evaluation completed at $(date)" >> $LOG_FILE
            # --batch_size 2 || { echo "1" > /tmp/process_status.$$; kill -TERM -$$; }

    ) &
done

# Wait for all processes to complete
echo "" | tee -a $LOG_FILE
echo "Waiting for all processes to complete..." | tee -a $LOG_FILE
wait

echo "All parallel processes completed at $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run final metrics calculation
echo "Starting metrics calculation..." | tee -a $LOG_FILE
python evaluation/calculate_iou_with_bbox.py --output_dir $OUTPUT_PATH 2>&1 | tee -a "$LOG_DIR/metrics_calculation.log"

echo "" | tee -a $LOG_FILE
echo "=================================" | tee -a $LOG_FILE
echo "Evaluation Completed: $(date)" | tee -a $LOG_FILE
echo "Log files saved in: $LOG_DIR" | tee -a $LOG_FILE
echo "=================================" | tee -a $LOG_FILE

