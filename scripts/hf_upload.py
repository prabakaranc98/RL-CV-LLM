"""Create (if needed) a public Hugging Face model repo and upload a folder to it.

    python scripts/hf_upload.py <user>/<repo> <folder>
Uses the token from `huggingface-cli login` (~/.cache/huggingface/token); never prints it.
"""
import sys

from huggingface_hub import HfApi

repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, private=False, exist_ok=True, repo_type="model")
api.upload_folder(repo_id=repo, folder_path=folder, commit_message="experimental WIP checkpoint (ex1-medbench-rl)")
print("uploaded", f"https://huggingface.co/{repo}")
