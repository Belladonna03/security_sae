import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer


plt.style.use("dark_background")

plt.rcParams.update({
    "figure.facecolor": "#0b0e1a",
    "axes.facecolor": "#0b0e1a",
    "savefig.facecolor": "#0b0e1a",

    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#e6e6e6",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "text.color": "#ffffff",

    "grid.color": "#2a2e45",
    "grid.alpha": 0.35,
    "grid.linestyle": "-",

    "font.size": 11,
    "axes.titleweight": "bold",
})

def bucket_source(source_dataset: str) -> str:
    s = (source_dataset or "").strip().lower()

    if s == "sharegpt":
        return "sharegpt"
    if s == "openwebtext":
        return "openwebtext"
    if s in ("chatgpt-jailbreak", "chatgpt_jailbreak", "jailbreak"):
        return "jlbrk"
    if s in ("advbench", "hh-rlhf", "hh_rlhf"):
        return "unsafe"
    return f"other:{s or 'unknown'}"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def serialize_messages(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in (messages or []))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_df(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in samples:
        source_dataset = s.get("source_dataset", "")
        b = bucket_source(source_dataset)

        text = serialize_messages(s.get("messages", []))
        rows.append(
            {
                "id": s.get("id"),
                "type": s.get("type"),
                "source_dataset": source_dataset,
                "bucket": b,
                "len_chars": len(text),
            }
        )
    return pd.DataFrame(rows)


def compute_token_counts(df: pd.DataFrame, samples: List[Dict[str, Any]], model_id: str, max_tokens: int) -> pd.Series:
    tok = AutoTokenizer.from_pretrained(model_id)
    token_counts = []
    for s in samples:
        text = serialize_messages(s.get("messages", []))
        enc = tok(text, truncation=True, max_length=max_tokens)
        token_counts.append(len(enc["input_ids"]))
    return pd.Series(token_counts, name="len_tokens")


def fig1_source_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["bucket"].value_counts()

    if counts.empty:
        raise RuntimeError("No data to plot (bucket counts empty). Check your JSONL path and format.")

    plt.figure(figsize=(9, 6))
    counts.plot(kind="bar")
    plt.title("Figure 1: Dataset Composition by Source (Buckets)")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)

    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{int(v)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig2_length_distribution(df: pd.DataFrame, out_path: Path) -> dict:
    lengths = df["len_chars"].astype(int).values

    total = int(len(lengths))
    mean = int(np.mean(lengths))
    median = int(np.median(lengths))
    mn = int(np.min(lengths))
    mx = int(np.max(lengths))

    p95 = int(np.percentile(lengths, 95))
    tail_n = int((lengths > p95).sum())
    tail_pct = tail_n / total * 100.0

    clipped = lengths[lengths <= p95]

    plt.figure(figsize=(12, 6))
    plt.hist(clipped, bins=80, color="#7aa2f7", alpha=0.9)

    if mean <= p95:
        plt.axvline(mean, linestyle="--", linewidth=2, color="#9ece6a", label=f"Mean = {mean}")
    if median <= p95:
        plt.axvline(median, linestyle=":", linewidth=2, color="#f7768e", label=f"Median = {median}")

    plt.title("Figure 2: Prompt Length Distribution (chars, clipped at p95)")
    plt.xlabel(f"Prompt length (chars) — shown up to p95 = {p95}")
    plt.ylabel("Number of prompts")
    plt.grid(alpha=0.3)
    plt.legend()

    stats = (
        f"Total: {total}\n"
        f"Mean: {mean}\n"
        f"Median: {median}\n"
        f"Min: {mn}\n"
        f"Max: {mx}\n"
        f"p95: {p95}\n"
        f"Tail > p95: {tail_n} ({tail_pct:.2f}%)"
    )
    plt.text(
        0.98, 0.95, stats,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {"total": total, "mean": mean, "median": median, "min": mn, "max": mx, "p95": p95}


def fig3_avg_length_by_bucket(df: pd.DataFrame, out_path: Path) -> None:
    avg = df.groupby("bucket")["len_chars"].mean().sort_values(ascending=False)

    plt.figure(figsize=(9, 6))
    avg.plot(kind="bar")
    plt.title("Figure 3: Average Prompt Length by Source (Buckets)")
    plt.ylabel("Average length (chars)")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)

    for i, v in enumerate(avg.values):
        plt.text(i, v, f"{int(round(v))}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_readme_stats(
    df: pd.DataFrame,
    out_md: Path,
    length_stats: Dict[str, int],
    token_shares: Optional[pd.DataFrame] = None,
) -> None:
    c = df["bucket"].value_counts()
    samples_pct = (c / c.sum() * 100).round(2)

    lines = []
    lines.append("## Dataset analysis (post token-budget)\n")

    lines.append("### Composition by samples\n")
    for k in c.index:
        lines.append(f"- **{k}**: {int(c[k])} ({samples_pct[k]}%)")
    lines.append("")

    lines.append("### Prompt length (chars)\n")
    lines.append(
        f"- total: {length_stats['total']}\n"
        f"- mean: {length_stats['mean']}\n"
        f"- median: {length_stats['median']}\n"
        f"- min: {length_stats['min']}\n"
        f"- max: {length_stats['max']}\n"
    )

    if token_shares is not None:
        lines.append("\n### Composition by tokens\n")
        for _, r in token_shares.iterrows():
            lines.append(f"- **{r['bucket']}**: {int(r['tokens'])} ({r['pct']:.2f}%)")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main(
    dataset_jsonl: str = "dataset/final_dataset.jsonl",
    out_dir: str = "assets/readme_figures",
    model_id_for_tokens: Optional[str] = None,
    max_tokens_per_example: int = 4096,
) -> None:
    out = Path(out_dir)
    ensure_dir(out)

    samples = load_jsonl(dataset_jsonl)
    df = build_df(samples)

    token_shares = None
    if model_id_for_tokens:
        df["len_tokens"] = compute_token_counts(df, samples, model_id_for_tokens, max_tokens_per_example)
        tok = df.groupby("bucket")["len_tokens"].sum().sort_values(ascending=False)
        token_shares = pd.DataFrame({"bucket": tok.index, "tokens": tok.values})
        token_shares["pct"] = token_shares["tokens"] / token_shares["tokens"].sum() * 100
        token_shares.to_csv(out / "token_shares.csv", index=False)

    fig1_source_distribution(df, out / "figure1_source_distribution.png")
    length_stats = fig2_length_distribution(df, out / "figure2_prompt_length_distribution.png")
    fig3_avg_length_by_bucket(df, out / "figure3_avg_length_by_source.png")

    df.to_csv(out / "dataset_summary.csv", index=False)
    write_readme_stats(df, out / "stats.md", length_stats, token_shares=token_shares)

    print(f"Saved 3 figures + stats into: {out.resolve()}")


if __name__ == "__main__":
    main(
        dataset_jsonl="dataset/final_dataset.jsonl",
        out_dir="assets/readme_figures",
        model_id_for_tokens="Qwen/Qwen3-4B",
        max_tokens_per_example=4096,
    )
