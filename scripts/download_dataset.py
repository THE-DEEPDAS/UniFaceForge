"""Download the gated UniFaceForge archives after Hugging Face authentication.

Usage: huggingface-cli login, then run this script with a destination path.
The archives total roughly 94 GB and are not unpacked or duplicated here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    p=argparse.ArgumentParser(); p.add_argument("destination",type=Path); args=p.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True)
    for repo in ("TheDeepDas/UniFaceForge","TheDeepDas/UniFaceForge-Real-Images"):
        out=snapshot_download(repo_id=repo,repo_type="dataset",local_dir=args.destination/repo.rsplit("/",1)[-1],resume_download=True)
        print(out)

if __name__ == "__main__": main()
