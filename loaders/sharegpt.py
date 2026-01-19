from datasets import load_dataset
from typing import Dict, Any, List
from collections import defaultdict

def load_sharegpt(split="train") -> List[Dict[str, Any]]:
    print("Loading ShareGPT dataset...")

    ds = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
        split=split
    )

    print(f"Total rows in raw dataset: {len(ds)}")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in ds:
        dialog_id = row.get("id")
        conversations = row.get("conversations", [])
        for msg in conversations:
            content = msg.get("value", "").strip()
            if not content:
                continue
            grouped[dialog_id].append({
                "from": msg.get("from"),
                "content": content
            })

    samples: List[Dict[str, Any]] = []

    for dialog_id, messages_raw in grouped.items():
        messages: List[Dict[str, str]] = []
        for msg in messages_raw:
            role = msg["from"]
            if role == "human":
                role = "human"
            elif role == "gpt":
                role = "assistant"
            else:
                continue

            messages.append({
                "role": role,
                "content": msg["content"]
            })

        sample = {
            "id": dialog_id,
            "type": "dialogue",
            "source_dataset": "sharegpt",
            "messages": messages,
            "metadata": {
                "jailbreak_score": None,
                "votes": None,
                "extra": {}
            }
        }

        samples.append(sample)

    print(f"Finished loading ShareGPT. Total valid samples: {len(samples)}")
    return samples
