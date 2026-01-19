from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase

class DatasetTokenizer:
    """
    Tokenizes chat-style datasets and applies per-example and total token limits.
    Returns fully prepared dataset с 'input_ids' и 'attention_mask'.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens_per_example: int,
        total_tokens_limit: int,
    ):
        self.tokenizer = tokenizer
        self.max_tokens_per_example = max_tokens_per_example
        self.total_tokens_limit = total_tokens_limit

    def _serialize_messages(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def tokenize_dataset(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Возвращает полностью токенизированный и фильтрованный по токенам датасет.
        Каждый пример содержит:
            - input_ids
            - attention_mask
            - исходные messages
        """
        filtered = []
        total_tokens = 0
        skipped_too_long = 0

        for s in samples:
            text = self._serialize_messages(s["messages"])
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_tokens_per_example,
            )

            n_tokens = len(enc["input_ids"])
            if n_tokens > self.max_tokens_per_example:
                skipped_too_long += 1
                continue

            if total_tokens + n_tokens > self.total_tokens_limit:
                break

            sample_copy = s.copy()
            sample_copy["input_ids"] = enc["input_ids"]
            sample_copy["attention_mask"] = enc.get("attention_mask")

            filtered.append(sample_copy)
            total_tokens += n_tokens

        print(
            f"Tokenized dataset: {len(filtered)} samples, "
            f"total tokens: {total_tokens}, "
            f"skipped too long: {skipped_too_long}"
        )
        
        return filtered
