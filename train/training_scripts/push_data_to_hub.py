import argparse
import os
from pathlib import Path
import inspect
import shutil
import tempfile

from datasets import DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi, create_repo


def load_any_dataset(data_path: str, split: str) -> DatasetDict:
    """
    Supported inputs:
    - A directory created by datasets.save_to_disk(): loaded via load_from_disk
    - A .json / .jsonl file: loaded via load_dataset("json", data_files=...)
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    if p.is_dir():
        ds = load_from_disk(str(p))
        if isinstance(ds, DatasetDict):
            return ds
        # Compatibility: a single Dataset (not DatasetDict)
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
        "--raw_only",
        action="store_true",
        help="Only upload the raw .json/.jsonl file itself (no datasets parsing, no parquet in data/).",
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
    parser.add_argument(
        "--upload_raw_json",
        action="store_true",
        help="When --data_path is a .json/.jsonl file, also upload the raw file itself to the repo.",
    )
    parser.add_argument(
        "--raw_path_in_repo",
        default="raw",
        help='Where to put the raw JSON file in the repo when using --upload_raw_json (default: "raw"). '
             'Use "" to upload to repo root.',
    )
    parser.add_argument(
        "--raw_commit_message",
        default="Upload raw file",
        help="Commit message when uploading a raw JSON/JSONL file.",
    )
    parser.add_argument("--split", default="train", help="Split name when uploading a single file/dataset")
    parser.add_argument("--private", action="store_true", help="Create/push as private dataset repo")
    args = parser.parse_args()

    repo_name = args.repo_name

    # Create (or ensure) the dataset repo exists
    create_repo(repo_name, repo_type="dataset", private=args.private, exist_ok=True)

    # Push to the Hub (requires prior login via `hf auth login` or setting HF_TOKEN)
    if args.raw_only:
        p = Path(args.data_path)
        if not (p.exists() and p.is_file() and p.suffix.lower() in {".json", ".jsonl"}):
            raise ValueError("--raw_only requires --data_path to be an existing .json/.jsonl file")

        api = HfApi()
        path_in_repo = args.raw_path_in_repo.strip()
        filename = p.name
        if path_in_repo:
            path_in_repo = f"{path_in_repo.rstrip('/')}/{filename}"
        else:
            path_in_repo = filename

        api.upload_file(
            repo_id=repo_name,
            repo_type="dataset",
            path_or_fileobj=str(p),
            path_in_repo=path_in_repo,
            commit_message=args.raw_commit_message,
        )
        print(f"Raw file uploaded to: {repo_name} ({path_in_repo})")
        return

    if args.folder_path:
        folder = Path(args.folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"--folder_path is not a directory: {args.folder_path}")

        api = HfApi()
        # For big directories, upload_large_folder is more reliable (uploads in smaller batches).
        if args.use_large_folder and hasattr(api, "upload_large_folder"):
            sig = inspect.signature(api.upload_large_folder)
            supports_path_in_repo = "path_in_repo" in sig.parameters
            supports_commit_message = "commit_message" in sig.parameters

            # Backward compatibility: older huggingface_hub may not support path_in_repo.
            if args.path_in_repo and not supports_path_in_repo:
                prefix = args.path_in_repo.strip().strip("/")
                with tempfile.TemporaryDirectory(prefix="hf_upload_") as tmpdir:
                    staged_root = Path(tmpdir)
                    staged_target = staged_root / prefix
                    staged_target.mkdir(parents=True, exist_ok=True)

                    # Prefer hardlinks to avoid copying; fallback to copying if hardlink fails.
                    for src_path in folder.rglob("*"):
                        rel = src_path.relative_to(folder)
                        dst_path = staged_target / rel
                        if src_path.is_dir():
                            dst_path.mkdir(parents=True, exist_ok=True)
                            continue
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(src_path, dst_path)
                        except Exception:
                            shutil.copy2(src_path, dst_path)

                    kwargs = dict(
                        repo_id=repo_name,
                        repo_type="dataset",
                        folder_path=str(staged_root),
                    )
                    if supports_commit_message:
                        kwargs["commit_message"] = args.commit_message
                    api.upload_large_folder(**kwargs)
            else:
                kwargs = dict(
                    repo_id=repo_name,
                    repo_type="dataset",
                    folder_path=str(folder),
                )
                if supports_commit_message:
                    kwargs["commit_message"] = args.commit_message
                if supports_path_in_repo:
                    kwargs["path_in_repo"] = args.path_in_repo
                api.upload_large_folder(**kwargs)
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

    # Optional: also upload the raw JSON/JSONL file itself (without datasets conversion).
    p = Path(args.data_path)
    if args.upload_raw_json and p.exists() and p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
        api = HfApi()
        path_in_repo = args.raw_path_in_repo.strip()
        filename = p.name
        if path_in_repo:
            path_in_repo = f"{path_in_repo.rstrip('/')}/{filename}"
        else:
            path_in_repo = filename

        api.upload_file(
            repo_id=repo_name,
            repo_type="dataset",
            path_or_fileobj=str(p),
            path_in_repo=path_in_repo,
            commit_message=f"Upload raw file: {filename}",
        )
        print(f"Raw file uploaded to: {repo_name} ({path_in_repo})")

    print(f"Dataset uploaded to: {repo_name}")
    print("\nDataset info:")
    print(dataset)


if __name__ == "__main__":
    main()