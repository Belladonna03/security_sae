from datasets import load_dataset
from typing import Dict, Any, List
import hashlib
import json

from .utils import make_id

def load_hh_rlhf(split="train") -> List[Dict[str, Any]]:
    dataset_name = "Anthropic/hh-rlhf"
    print(f"Loading dataset {dataset_name} (split={split})...")

    ds = load_dataset(dataset_name, split=split)
    print(f"Total rows in raw dataset: {len(ds)}")

    samples: List[Dict[str, Any]] = []

    for i, row in enumerate(ds, start=1):

        chosen = row.get("chosen")
        rejected = row.get("rejected")

        if not chosen or not rejected:
            continue

        messages = [
            {"role": "assistant", "content": chosen.strip()},
            {"role": "assistant", "content": rejected.strip()},
        ]

        if any(not m["content"] for m in messages):
            continue

        sample = {
            "id": None,
            "type": "preference_pair",
            "source_dataset": "hh-rlhf",
            "messages": messages,
            "metadata": {
                "jailbreak_score": None,
                "votes": None,
                "extra": {
                    "preference": "chosen"
                }
            }
        }

        sample["id"] = make_id(sample)
        samples.append(sample)

    print(f"Finished loading {dataset_name}. Total valid samples: {len(samples)}")
    return samples
