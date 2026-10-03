"""Reproduce the local dataset downloads used by the resource-finder phase."""

from pathlib import Path
import argparse
import requests
import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent


FILES = {
    "Hello-SimpleAI/HC3": ["all.jsonl", "README.md"],
    "browndw/human-ai-parallel-corpus": [
        "README.md",
        "text_data/hape-text_gpt-4o-2024-08-06.parquet",
        "text_data/hape-text_gpt-4o-mini-2024-07-18.parquet",
        "text_data/hape-text_human-chunk-1.parquet",
        "text_data/hape-text_human-chunk-2.parquet",
        "text_data/hape-text_llama-3-70B-Instruct.parquet",
        "text_data/hape-text_llama-3-70B.parquet",
        "text_data/hape-text_llama-3-8B-Instruct.parquet",
        "text_data/hape-text_llama-3-8B.parquet",
    ],
    "yaful/MAGE": [
        "README.md", "train.csv", "valid.csv", "test.csv",
        "test_ood_set_gpt.csv", "test_ood_set_gpt_para.csv",
        "prepare_testbeds.py",
    ],
}


def download_standard() -> None:
    for repo, filenames in FILES.items():
        local = ROOT / repo.split("/")[-1]
        for filename in filenames:
            hf_hub_download(
                repo_id=repo, repo_type="dataset", filename=filename,
                local_dir=local,
            )


def download_raid_sample() -> None:
    """Fetch evenly spaced blocks without downloading RAID's 16.7 GB CSVs."""
    out = ROOT / "raid"
    out.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="liamdugan/raid", repo_type="dataset",
        filename="README.md", local_dir=out,
    )
    rows = []
    for offset in [round(i * (2_269_900 / 47)) for i in range(48)]:
        response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": "liamdugan/raid", "config": "raid",
                "split": "train", "offset": offset, "length": 100,
            },
            timeout=90,
        )
        response.raise_for_status()
        rows.extend(item["row"] for item in response.json()["rows"])
    pd.DataFrame(rows).drop_duplicates("id").to_parquet(
        out / "train_sample.parquet", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raid-full", action="store_true",
        help="Download the complete RAID train/extra files (about 16.7 GB).",
    )
    args = parser.parse_args()
    download_standard()
    if args.raid_full:
        from datasets import load_dataset
        load_dataset("liamdugan/raid").save_to_disk(str(ROOT / "raid_full"))
    else:
        download_raid_sample()
