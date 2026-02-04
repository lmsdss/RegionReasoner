#!/bin/bash
# -*- coding: utf-8 -*-
set -e
export HF_ENDPOINT=https://hf-mirror.com

MODEL_TYPE="vision_reasoner"  # Model type: qwen or vision_reasoner or qwen2
TEST_DATA_PATH=${1:-"Ricky06662/refcoco_val"}

# Extract model name and test dataset name for output directory
TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
# OUTPUT_PATH="/sunwenfang/VisionReasoner/detection_eval_results/${MODEL_TYPE}/${TEST_NAME}"
OUTPUT_PATH="/sunwenfang/VisionReasoner/detection_eval_results/${TEST_NAME}"
# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/8a100611/global_step_443/actor/huggingface" # --model_path $MODEL_PATH \
# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/2h20klloss/global_step_443/actor/huggingface"
# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/2h20cliploss/global_step_443/actor/huggingface"
# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/ydu/global_step_443/actor/huggingface"
# MODEL_PATH="/sunwenfang/VisionReasoner/pretrained_models/VisionReasoner-7B"

# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/run_visionreasoner_7b_local_json_20250819_18/global_step_443/actor/huggingface“

# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/run_visionreasoner_7b_local_json_20250820_15/global_step_443/actor/huggingface"

# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/visionreasoner_7b_split1999_20250821_17/global_step_551/actor/huggingface"


# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/run_visionreasoner_7b_splitfull_reward2/global_step_658/actor/huggingface"
# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/run_visionreasoner_7b_splitfull_reward2_1_0/global_step_600/actor/huggingface"

# MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/run_visionreasoner_7b_splitfull_reward1/global_step_658/actor/huggingface"

MODEL_PATH="/sunwenfang/segzero/visionmanus_workdir/prompt5_reward2_1_0_para_0d_3_5_2_5_8gpu/global_step_658/actor/huggingface"

echo $MODEL_PATH

# Customize GPU array here - specify which GPUs to use
# GPU_ARRAY=(0 1 2 3 4 5 6 7)  # Example: using GPUs 0, 1, 2, 3
GPU_ARRAY=(0 1 2 3)  # Example: using GPUs 0, 1, 2, 3

NUM_PARTS=${#GPU_ARRAY[@]}

# Create output directory
mkdir -p $OUTPUT_PATH

# Run processes in parallel
for i in $(seq 0 $((NUM_PARTS-1))); do
    gpu_id=${GPU_ARRAY[$i]}
    process_idx=$i  # 0-based indexing for process
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    (
        python evaluation/evaluation_segmentation.py \
            --model $MODEL_TYPE \
            --model_path $MODEL_PATH \
            --output_path $OUTPUT_PATH \
            --test_data_path $TEST_DATA_PATH \
            --idx $process_idx \
            --num_parts $NUM_PARTS \
            --batch_size 2 
            # --batch_size 2 || { echo "1" > /tmp/process_status.$$; kill -TERM -$$; }

    ) &
done

# Wait for all processes to complete
wait

python evaluation/calculate_iou_with_bbox.py --output_dir $OUTPUT_PATH
