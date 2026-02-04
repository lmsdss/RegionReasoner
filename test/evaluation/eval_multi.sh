#!/bin/bash
set -e

MODEL_TYPE="vision_reasoner"  # Model type: qwen or vision_reasoner or qwen2

# Resolve paths relative to this script so the repo can be cloned and run anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"     # .../RegionReasoner/test
REPO_ROOT="$(cd "${TEST_ROOT}/.." && pwd)"     # .../RegionReasoner

usage() {
    echo "Usage: $0 [TEST_DATA_PATH] [MODEL_PATH]"
    echo ""
    echo "Arguments:"
    echo "  TEST_DATA_PATH    Path to a test JSON file (absolute path) OR a filename under test/testdata/"
    echo "  MODEL_PATH        Hugging Face model id (e.g. lmsdss/RegionReasoner-7B) or a local model directory"
    echo ""
    echo "Examples:"
    echo "  $0 refcocog_multi_turn.json lmsdss/RegionReasoner-7B"
    echo "  $0 ${TEST_ROOT}/testdata/refcocoplus_multi_turn.json /path/to/local/model_dir"
    echo ""
    echo "Available test files under: ${TEST_ROOT}/testdata/"
    ls -1 "${TEST_ROOT}/testdata/"*.json 2>/dev/null | xargs -n1 basename || echo "No JSON files found."
    exit 1
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided, use defaults
    TEST_DATA_PATH="${TEST_ROOT}/testdata/refcocog_multi_turn.json"
    MODEL_PATH="lmsdss/RegionReasoner-7B"
    echo "No arguments provided, using defaults:"
    echo "  - Test data: $(basename "$TEST_DATA_PATH")"
    echo "  - Model: $MODEL_PATH"
elif [ $# -eq 1 ]; then
    # One argument provided (test data path)
    if [[ "$1" == /* ]]; then
        # Absolute path provided
        TEST_DATA_PATH="$1"
    else
        # Relative filename provided, resolve under repo
        TEST_DATA_PATH="${TEST_ROOT}/testdata/$1"
    fi
    # Use default model path
    MODEL_PATH="lmsdss/RegionReasoner-7B"
    echo "Using test data: $(basename $TEST_DATA_PATH)"
    echo "Using default model: $MODEL_PATH"
elif [ $# -eq 2 ]; then
    # Two arguments provided (test data path and model path)
    if [[ "$1" == /* ]]; then
        TEST_DATA_PATH="$1"
    else
        TEST_DATA_PATH="${TEST_ROOT}/testdata/$1"
    fi
    
    # Parse model path argument
    case "$2" in
        "pretrained")
            MODEL_PATH="lmsdss/RegionReasoner-7B"
            echo "Using pretrained RegionReasoner-7B model (from Hugging Face Hub)"
            ;;
        "trained")
            if [[ -z "${TRAINED_MODEL_PATH:-}" ]]; then
                echo "Error: 'trained' shortcut requires TRAINED_MODEL_PATH to be set."
                echo "Example: export TRAINED_MODEL_PATH=/path/to/your/trained_model_dir"
                exit 1
            fi
            MODEL_PATH="${TRAINED_MODEL_PATH}"
            echo "Using fine-tuned model from TRAINED_MODEL_PATH"
            ;;
        /*)
            # Absolute path provided
            MODEL_PATH="$2"
            echo "Using custom model path: $MODEL_PATH"
            ;;
        *)
            # Treat as a relative path from repo root
            echo "Warning: Unrecognized model shortcut '$2', treating as path relative to repo root"
            MODEL_PATH="${REPO_ROOT}/$2"
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
    echo "Available files under ${TEST_ROOT}/testdata/:"
    ls -la "${TEST_ROOT}/testdata/"*.json 2>/dev/null | grep -E '\.(json)$' || echo "No JSON files found"
    exit 1
fi

# Validate local model path if it looks like a local directory path
if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* || "$MODEL_PATH" == ../* ]]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Model directory not found: $MODEL_PATH"
        exit 1
    fi
fi

# Extract filename for output directory naming
TEST_FILENAME=$(basename "$TEST_DATA_PATH" .json)
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_PATH="${TEST_ROOT}/detection_eval_results/test_multi/${TEST_FILENAME}_${MODEL_NAME}_$(date +%m%d%H%M)"

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
GPU_ARRAY=(0 1 2 3)  # Example: using GPUs 0, 1, 2, 3

NUM_PARTS=${#GPU_ARRAY[@]}

echo "Using GPUs: ${GPU_ARRAY[*]}" | tee -a $LOG_FILE
echo "Number of parallel processes: $NUM_PARTS" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Create output directory
mkdir -p $OUTPUT_PATH

export HF_ENDPOINT=https://hf-mirror.com

# Ensure relative paths work regardless of where the script is invoked.
cd "${TEST_ROOT}"

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