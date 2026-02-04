#!/usr/bin/env python3
import argparse
import inspect
import io
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def push_to_hub(
    model_path: str,
    repo_name: str,
    private: bool = False,
    use_large_folder: bool = False,
    commit_message: str = "Upload model files",
):
    """
    Push a local model directory to the Hugging Face Hub.
    
    Args:
        model_path: Local model directory (e.g., contains config/tokenizer/weights files).
        repo_name: Hugging Face repo id (format: username/repo-name).
        private: Create/push as a private repo.
        use_large_folder: Use upload_large_folder for very large directories (more robust).
        commit_message: Commit message (only effective if the API supports it).
    """
    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"model_path is not a directory: {model_path}")

    # Initialize Hugging Face Hub API
    api = HfApi()
    
    # Create the repo (or ensure it exists)
    try:
        create_repo(repo_name, repo_type="model", private=private, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Create a minimal model card
    readme_content = """# RegionReasoner-7B

This repository contains model files for RegionReasoner.

- Code: https://github.com/lmsdss/RegionReasoner
"""

    # Upload README first (avoid copying the whole model dir)
    try:
        api.upload_file(
            repo_id=repo_name,
            repo_type="model",
            path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
            path_in_repo="README.md",
            commit_message="Update README",
        )
    except Exception as e:
        # README upload failure should not block weight uploads
        print(f"Failed to upload README.md (can be ignored): {e}")

    # Upload model directory
    if use_large_folder and hasattr(api, "upload_large_folder"):
        sig = inspect.signature(api.upload_large_folder)
        kwargs = {
            "repo_id": repo_name,
            "repo_type": "model",
            "folder_path": str(model_dir),
        }
        # Backward compatibility: older upload_large_folder may not support commit_message/private
        if "commit_message" in sig.parameters:
            kwargs["commit_message"] = commit_message
        if "private" in sig.parameters:
            kwargs["private"] = private
        api.upload_large_folder(**kwargs)
    else:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message,
        )

    print(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")

def main():
    parser = argparse.ArgumentParser(description="Push a local model directory to Hugging Face Hub.")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Local model directory path")
    parser.add_argument("--repo_name", type=str, required=True,
                      help="Hugging Face repo id (format: username/repo-name)")
    parser.add_argument("--private", action="store_true",
                      help="Create/push as a private repo")
    parser.add_argument("--use_large_folder", action="store_true",
                      help="Use upload_large_folder for very large directories (more robust)")
    parser.add_argument("--commit_message", type=str, default="Upload model files",
                      help="Commit message (only effective if the API supports it)")

    args = parser.parse_args()
    
    push_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
        use_large_folder=args.use_large_folder,
        commit_message=args.commit_message,
    )

# Usage:
# python push_model_to_hub.py \
#     --model_path "/path/to/your/model" \
#     --repo_name "your-username/model-name"
if __name__ == "__main__":
    main()

