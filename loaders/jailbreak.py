from datasets import load_dataset
from typing import Dict, Any, List
import hashlib
import json

from .utils import make_id


def load_chatgpt_jailbreak(split="train") -> List[Dict[str, Any]]:
    dataset_name = "rubend18/ChatGPT-Jailbreak-Prompts"
    print(f"Loading dataset {dataset_name} (split={split})...")

    ds = load_dataset(dataset_name, split=split)
    print(f"Total rows in raw dataset: {len(ds)}")

    samples: List[Dict[str, Any]] = []

    for i, row in enumerate(ds, start=1):

        prompt = row.get("Prompt", "")
        if not prompt:
            continue
        prompt = prompt.strip()
        if not prompt:
            continue

        sample = {
            "id": None,
            "type": "dialogue",
            "source_dataset": "chatgpt-jailbreak",
            "messages": [
                {"role": "human", "content": prompt}
            ],
            "metadata": {
                "jailbreak_score": row.get("Jailbreak Score"),
                "votes": row.get("Votes"),
                "extra": {
                    "attack_type": row.get("Name", "unknown"),
                    "tested_on_gpt4": row.get("GPT-4")
                }
            }
        }

        sample["id"] = make_id(sample)
        samples.append(sample)

    print(f"Finished loading {dataset_name}. Total valid samples: {len(samples)}")
    return samples
