import json
import random
from typing import Dict, List

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from loaders.sharegpt import load_sharegpt
from loaders.openwebtext import load_openwebtext
from loaders.jailbreak import load_chatgpt_jailbreak
from loaders.advbench import load_advbench
from loaders.hh_rlhf import load_hh_rlhf


class DatasetMixer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens_per_example: int,
        total_tokens_limit: int,
        seed: int,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_example = max_tokens_per_example
        self.total_tokens_limit = total_tokens_limit
        self.seed = seed

    @staticmethod
    def load_all() -> Dict[str, List[dict]]:
        return {
            "sharegpt": load_sharegpt(),
            "openwebtext": load_openwebtext(),
            "jlbrk": load_chatgpt_jailbreak(),
            "unsafe": load_advbench() + load_hh_rlhf(),
        }

    def mix(
        self,
        datasets: Dict[str, List[dict]],
        total_samples: int,
    ) -> List[dict]:
        random.seed(self.seed)

        proportions = {
            "sharegpt": 0.6,
            "openwebtext": 0.2,
            "jlbrk": 0.1,
            "unsafe": 0.1,
        }

        mixed: List[dict] = []

        for name, ratio in proportions.items():
            target = int(total_samples * ratio)
            pool = datasets[name]

            if len(pool) < target:
                target = len(pool)

            mixed.extend(random.sample(pool, target))

        random.shuffle(mixed)
        return mixed

    def _serialize_messages(self, sample: dict) -> str:
        return "\n".join(
            f"{m['role']}: {m['content']}"
            for m in sample["messages"]
        )

    def count_tokens(self, sample: dict) -> int:
        text = self._serialize_messages(sample)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens_per_example,
        )
        return len(encoded["input_ids"])

    def apply_token_budget(self, samples: List[dict]) -> List[dict]:
        filtered: List[dict] = []
        total_tokens = 0

        for sample in samples:
            n_tokens = self.count_tokens(sample)

            if total_tokens + n_tokens > self.total_tokens_limit:
                break

            filtered.append(sample)
            total_tokens += n_tokens

        return filtered

    @staticmethod
    def save_jsonl(samples: List[dict], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def save_hf(samples: List[dict], path: str) -> None:
        Dataset.from_list(samples).save_to_disk(path)
