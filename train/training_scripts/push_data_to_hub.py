import argparse
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi, create_repo


def load_any_dataset(data_path: str, split: str) -> DatasetDict:
    """
    支持两种输入：
    - datasets.save_to_disk() 生成的目录：load_from_disk
    - .json / .jsonl 文件：load_dataset("json", data_files=...)
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    if p.is_dir():
        ds = load_from_disk(str(p))
        if isinstance(ds, DatasetDict):
            return ds
        # 兼容单个 Dataset
        return DatasetDict({split: ds})

    suffix = p.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files={split: str(p)})

    raise ValueError(f"Unsupported data_path: {data_path} (expect dir, .json, or .jsonl)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a local dataset to HuggingFace Hub (dataset repo).")
    parser.add_argument("--repo_name", default="lmsdss/regionreasoner_data", help="e.g. username/my_dataset")
    parser.add_argument(
        "--data_path",
        default="RegionReasoner/train/data/regionreasoner_train.json",
        help="Path to dataset dir (save_to_disk) or a .json/.jsonl file",
    )
    parser.add_argument(
        "--folder_path",
        default=None,
        help="If set, upload the entire folder as files (no dataset parsing).",
    )
    parser.add_argument(
        "--path_in_repo",
        default="",
        help='Target subdir in repo when using --folder_path (e.g. "regionreasoner_data"). Default: repo root.',
    )
    parser.add_argument(
        "--commit_message",
        default="Upload folder",
        help="Commit message when using --folder_path.",
    )
    parser.add_argument(
        "--use_large_folder",
        action="store_true",
        help="Use HfApi.upload_large_folder for big directories (recommended).",
    )
    parser.add_argument("--split", default="train", help="Split name when uploading a single file/dataset")
    parser.add_argument("--private", action="store_true", help="Create/push as private dataset repo")
    args = parser.parse_args()

    repo_name = args.repo_name

    # 创建（或确保存在）dataset repo
    create_repo(repo_name, repo_type="dataset", private=args.private, exist_ok=True)

    # 推送到 Hub（需要提前登录：huggingface-cli login 或设置 HF_TOKEN）
    if args.folder_path:
        folder = Path(args.folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"--folder_path is not a directory: {args.folder_path}")

        api = HfApi()
        # 大目录更稳：分批多次提交，避免一次性上传过大而失败
        if args.use_large_folder and hasattr(api, "upload_large_folder"):
            api.upload_large_folder(
                repo_id=repo_name,
                repo_type="dataset",
                folder_path=str(folder),
                path_in_repo=args.path_in_repo,
                commit_message=args.commit_message,
            )
        else:
            api.upload_folder(
                repo_id=repo_name,
                repo_type="dataset",
                folder_path=str(folder),
                path_in_repo=args.path_in_repo,
                commit_message=args.commit_message,
            )
        print(f"Folder uploaded to: {repo_name} (path_in_repo={args.path_in_repo!r})")
        return

    dataset = load_any_dataset(args.data_path, args.split)
    dataset.push_to_hub(repo_name, private=args.private)

    print(f"Dataset uploaded to: {repo_name}")
    print("\nDataset info:")
    print(dataset)


if __name__ == "__main__":
    main()