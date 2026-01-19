from datasets import load_dataset
from typing import Dict, Any, List
import os

from .utils import make_id

def load_advbench(split="train") -> List[Dict[str, Any]]:
    dataset_name = "walledai/AdvBench"
    print(f"Loading dataset {dataset_name} (split={split})...")

    ds = load_dataset(dataset_name, split=split)
    print(f"Total rows in raw dataset: {len(ds)}")

    samples: List[Dict[str, Any]] = []

    for i, row in enumerate(ds, start=1):

        prompt = row.get("prompt", "").strip()
        if not prompt:
            continue

        sample = {
            "id": None,
            "type": "dialogue",
            "source_dataset": "advbench",
            "messages": [{"role": "human", "content": prompt}],
            "metadata": {
                "jailbreak_score": None,
                "votes": None,
                "extra": {
                    "category": row.get("category", "unknown"),
                    "expected_behavior": "refusal",
                    "is_unsafe": True
                }
            }
        }

        sample["id"] = make_id(sample)
        samples.append(sample)

    print(f"Finished loading {dataset_name}. Total valid samples: {len(samples)}")
    return samples
