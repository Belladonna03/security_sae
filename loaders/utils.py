import hashlib
import json
from pathlib import Path
from datasets import Dataset
from typing import List, Dict, Any


def make_id(sample: dict) -> str:
    core = {
        "type": sample["type"],
        "messages": sample.get("messages"),
        "text": sample.get("text"),
    }
    raw = json.dumps(core, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Загрузка готового JSONL датасета."""
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def save_jsonl(samples: List[Dict[str, Any]], path: str):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved JSONL to {path}")

def save_hf_dataset(samples: List[Dict[str, Any]], path: str):
    ds = Dataset.from_list(samples)
    ds.save_to_disk(path)
    print(f"Saved HF dataset to {path}")
