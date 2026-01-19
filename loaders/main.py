from transformers import AutoTokenizer
from loaders.sharegpt import load_sharegpt
from loaders.openwebtext import load_openwebtext
from loaders.jailbreak import load_chatgpt_jailbreak
from loaders.advbench import load_advbench
from loaders.hh_rlhf import load_hh_rlhf

from .dataset_mixer import DatasetMixer
from .tokenization import DatasetTokenizer
from .utils import load_jsonl, save_jsonl, save_hf_dataset


if __name__ == "__main__":
    RANDOM_SEED = 2026
    MODEL_ID = "Qwen/Qwen3-4B"

    MAX_TOKENS_PER_EXAMPLE = 4096
    TOTAL_TOKENS_LIMIT = 100_000_000
    TOTAL_SAMPLES = 100_000
    DATASET_JSONL_PATH = "dataset/final_dataset.jsonl"
    DATASET_HF_PATH = "final_dataset_hf"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    mixer = DatasetMixer(
        tokenizer=tokenizer,
        max_tokens_per_example=MAX_TOKENS_PER_EXAMPLE,
        total_tokens_limit=TOTAL_TOKENS_LIMIT,
        seed=RANDOM_SEED,
    )

    datasets = mixer.load_all()
    mixed = mixer.mix(datasets, TOTAL_SAMPLES)
    final_dataset = mixer.apply_token_budget(mixed)

    mixer.save_jsonl(final_dataset, DATASET_JSONL_PATH)
    mixer.save_hf(final_dataset, DATASET_HF_PATH)

    dt = DatasetTokenizer(tokenizer, MAX_TOKENS_PER_EXAMPLE, TOTAL_TOKENS_LIMIT)
    dataset = load_jsonl(DATASET_JSONL_PATH)
    
    tokenized_dataset = dt.tokenize_dataset(dataset)

    save_jsonl(tokenized_dataset, "dataset/final_dataset_tokenized.jsonl")
    save_hf_dataset(tokenized_dataset, "final_dataset_tokenized_hf")
