from datasets import load_dataset
from typing import Dict, Any, List
from itertools import islice
import hashlib
import json

from .utils import make_id

def load_openwebtext(
    split: str = "train",
    limit: int = 10_000,
    min_length: int = 200,
) -> List[Dict[str, Any]]:
    """
    Стримит OpenWebText, берёт первые limit документов,
    фильтрует короткие тексты и генерирует ID.
    """

    dataset_name = "Skylion007/openwebtext"

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )

    samples: List[Dict[str, Any]] = []

    for i, row in enumerate(islice(ds, limit), start=1):

        text = row.get("text", "")
        if not text:
            continue

        text = text.strip()
        if len(text) < min_length:
            continue

        sample = {
            "id": None,
            "type": "text",
            "source_dataset": "openwebtext",
            "messages": [{"role": "human", "content": text}],
            "metadata": {
                "jailbreak_score": None,
                "votes": None,
                "extra": {},
            },
        }

        sample["id"] = make_id(sample)
        samples.append(sample)

    print(
        f"Finished loading {dataset_name}. "
        f"Requested: {limit}, valid samples: {len(samples)}"
    )
    return samples