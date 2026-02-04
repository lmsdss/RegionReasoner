# Launch RegionReasoner training.
bash /RegionReasoner/train/training_scripts/run_regionreasoner.sh


# NOTE: Replace /your/model/path/ with the training output model directory (e.g., the actor/ folder under a workdir) to merge/export a HuggingFace-format model.
python3 /RegionReasoner/train/training_scripts/model_merger.py --local_dir /your/model/path/


