export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export USE_WANDB=false

TIMESTAMP=$(date '+%Y%m%d_%H')  



MODEL_PATH="pretrained_models/Qwen2.5-VL-7B-Instruct" # replace it with your local qwen2.5-vl-7b-instruct file path

RUN_NAME=$(basename "$0" .sh)_${TIMESTAMP}


mkdir -p logs
echo "=================================================="
echo "🚀 TRAINING START: $(date)"
echo "📝 Experiment: $RUN_NAME"
echo "🏷️  Run Name: $RUN_NAME"
echo "=================================================="
python3 -m verl.trainer.main \
    config=training_scripts/regionreasoner_7b.yaml \
    data.train_files=regionreasoner_data/regionreasoner_train.json \
    data.val_files=None \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.global_batch_size=16   \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.reward.compute_score=region_reasoner \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=regionreasoner_workdir/00000000  \
    2>&1 | tee logs/00000000.log


echo "=================================================="
echo "✅ TRAINING COMPLETED: $(date)"
echo "📝 Experiment: $RUN_NAME"
echo "💾 Checkpoint: regionreasoner_workdir/${RUN_NAME}"
echo "=================================================="



