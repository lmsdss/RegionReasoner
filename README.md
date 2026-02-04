# RegionReasoner: Region-Grounded Multi-Round Visual Reasoning (ICLR 2026)

**RegionReasoner** is a reinforcement learning framework for grounded multi-round visual reasoning, coupled with **RegionDial-Bench** (detection + segmentation), that requires reasoning traces to cite reference bboxes and uses a global–local consistency reward to improve iterative reasoning accuracy and spatial grounding.

- **Paper**: [📖 RegionReasoner](https://arxiv.org/pdf/2602.03733)
- **Model**: [🤗 `lmsdss/RegionReasoner-7B`](https://huggingface.co/lmsdss/RegionReasoner-7B)
- **Training dataset**: [🤗 `lmsdss/regionreasoner_data`](https://huggingface.co/datasets/lmsdss/regionreasoner_data)
- **Test dataset**: [🤗 `lmsdss/regionreasoner_test_data`](https://huggingface.co/datasets/lmsdss/regionreasoner_test_data)


## Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start (Evaluation)](#quick-start-evaluation)
- [Training](#training)
- [Data](#data)
- [Model](#model)
- [Citation](#citation)

## Installation

This repo provides separate environments for `train/` and `test/`.

```bash
# Clone
git clone https://github.com/lmsdss/RegionReasoner.git
cd RegionReasoner
```

### Training environment

```bash
cd train
conda create -n RegionReasoner python=3.12
conda activate RegionReasoner
pip install torch==2.6.0 torchvision==0.21.0
pip install -e .
```

### Download training backbone (Git LFS)

RegionReasoner training uses a vision-language backbone (e.g., Qwen2.5-VL). Make sure `git-lfs` is available, then:

```bash
mkdir -p pretrained_models
cd pretrained_models
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
```

### Evaluation environment

```bash
cd ../test
conda create -n RegionReasoner_test python=3.12
conda activate RegionReasoner_test
pip3 install torch torchvision
pip install -r requirements.txt
```

### Download evaluation models (Git LFS)

```bash
cd pretrained_models
git lfs install
git clone https://huggingface.co/lmsdss/RegionReasoner-7B
git clone https://huggingface.co/Ricky06662/TaskRouter-1.5B
```


## Project Structure

```text
RegionReasoner/
  train/                      # training (RL) utilities, reward functions, scripts
    training_scripts/
    verl/
  test/                       # evaluation/inference utilities
    evaluation/
    testdata/
    pretrained_models/
    vision_reasoner/
  train.sh                    # convenience wrapper for training + model export
  test.sh                     # convenience wrapper for evaluation
```

## Quick Start (Evaluation)

### 1) Prepare test data

Example JSON files are under `test/testdata/` (e.g., multi-turn question files). Images are also placed under `test/testdata/` depending on the benchmark packaging.

### 2) Run multi-turn evaluation

From `RegionReasoner/test/`:

```bash
cd test

# Evaluate with a Hugging Face model id
bash evaluation/eval_multi.sh refcocog_multi_turn.json lmsdss/RegionReasoner-7B

# Or evaluate with a local model directory (must contain config/tokenizer/weights)
# bash evaluation/eval_multi.sh refcocog_multi_turn.json /path/to/local/model_dir
```

Outputs (logs + metrics + optional visualizations) will be written under:

```text
test/detection_eval_results/test_multi/
```

## Training

Training scripts live under `train/`. A typical workflow is:

1) **Run training**

```bash
cd train
bash training_scripts/run_regionreasoner.sh
```

2) **Merge/export the trained checkpoint to Hugging Face format**

```bash
# See train.sh for an example wrapper
bash ../train.sh

# Or run directly:
python3 training_scripts/model_merger.py --local_dir /path/to/your/workdir/.../actor
```

## Data

This repo assumes multi-turn evaluation JSON files with fields similar to:

- `image_id`
- `image_path`
- `conversational_turns` (a list of turns containing `question`, and optionally turn-level annotations such as `bboxes`, `masks`, etc.)

To make the test set easy to edit, we generate the final multi-turn file (e.g., `refcocoplus_multi_turn.json`) by merging questions from `refcocoplus_multi_question.json` with labels from `refcocoplus_multi_mask.json` using `test/merge_mask_json.py`.

Example (run from the repo root):

```bash
python test/merge_mask_json.py \
  --question_json test/testdata/refcocoplus_multi_question.json \
  --mask_json test/testdata/refcocoplus_multi_mask.json \
  --output_json test/testdata/refcocoplus_multi_turn.json
```


## Model

- Official released model: [🤗 `lmsdss/RegionReasoner-7B`](https://huggingface.co/lmsdss/RegionReasoner-7B)

## Acknowledgement

This project is built upon prior open-source efforts including [VisionReasoner](https://github.com/JIA-Lab-research/VisionReasoner/tree/main), [Seg-Zero](https://github.com/dvlab-research/Seg-Zero), [EasyR1](https://github.com/hiyouga/EasyR1), and [veRL](https://github.com/volcengine/verl).


## Citation

If you find this project useful, please cite our paper:

```bibtex
@article{regionreasoner2026,
  title        = {RegionReasoner: Region-Grounded Multi-Round Visual Reasoning},
  author       = {Wenfang Sun, Hao Chen, Yingjun Du, Yefeng Zheng, Cees G. M. Snoek},
  journal      = {International Conference on Learning Representations (ICLR)},
  year         = {2026},
  url          = {https://arxiv.org/pdf/2602.03733}
}
```
