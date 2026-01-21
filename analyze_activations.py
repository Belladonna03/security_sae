import os
import glob
import json
import random
import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=str, default="./sae_feats_layer20")
    ap.add_argument("--sample_tokens", type=int, default=20000)
    args = ap.parse_args()

    meta_path = os.path.join(args.dump_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {args.dump_dir}")

    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    pt_files = sorted(glob.glob(os.path.join(args.dump_dir, "layer*_chunk*.pt")))
    if not pt_files:
        raise SystemExit(
            f"No chunk files found in {args.dump_dir}. Check --dump_dir and filename pattern."
        )

    obj0 = torch.load(pt_files[0], map_location="cpu")
    if "sae_feats" not in obj0:
        raise ValueError(
            f"Expected 'sae_feats' key in chunk file. Got keys: {list(obj0.keys())}"
        )
    M = int(meta.get("SAE_D_FEAT", obj0["sae_feats"].shape[1]))

    count_tokens = 0
    sum_all = torch.zeros(M, dtype=torch.float64)
    sum_active = torch.zeros(M, dtype=torch.float64)
    active_count = torch.zeros(M, dtype=torch.int64)
    max_all = torch.zeros(M, dtype=torch.float32)
    total_active_entries = 0

    for f in pt_files:
        obj = torch.load(f, map_location="cpu")
        if "sae_feats" not in obj:
            raise ValueError(f"Missing 'sae_feats' in {f}")
        a = obj["sae_feats"].float()
        assert len(a.shape) == 2 and a.shape[1] == M, \
            f"Expected shape [N, {M}], got {a.shape}"

        count_tokens += a.shape[0]
        sum_all += a.sum(dim=0).double()
        is_active = a > 0
        active_count += is_active.sum(dim=0).to(torch.int64)
        sum_active += (a * is_active).sum(dim=0).double()
        max_all = torch.maximum(max_all, a.max(dim=0).values)
        total_active_entries += int(is_active.sum().item())

    mean_all = (sum_all / max(count_tokens, 1)).float()
    firing_rate = (active_count.float() / max(count_tokens, 1))
    mean_when_active = (sum_active / active_count.clamp_min(1).double()).float()

    out_npz = os.path.join(args.dump_dir, "feature_stats.npz")
    np.savez(
        out_npz,
        firing_rate=firing_rate.numpy(),
        mean_all=mean_all.numpy(),
        mean_when_active=mean_when_active.numpy(),
        max_all=max_all.numpy(),
        active_count=active_count.numpy(),
        token_count=np.array([count_tokens], dtype=np.int64),
        avg_active_feats_per_token=np.array([total_active_entries / max(count_tokens, 1)], dtype=np.float64),
    )
    print(f"Saved: {out_npz}")

    rng = random.Random(0)
    left = args.sample_tokens
    samples = []
    for f in pt_files:
        if left <= 0:
            break
        obj = torch.load(f, map_location="cpu")
        if "sae_feats" not in obj:
            continue
        a = obj["sae_feats"].float()
        n = a.shape[0]
        if n == 0:
            continue
        take = min(left, max(1, n // 50))
        take = min(take, n)
        idx = rng.sample(range(n), k=take)
        samples.append(a[idx])
        left -= take

    if not samples:
        print("Warning: no samples collected for sparsity report")
        return

    X = torch.cat(samples, dim=0)
    active_counts = (X > 0).sum(dim=1).numpy()
    print(f"Sampled tokens: {X.shape[0]}")
    print(
        f"Active feats/token - mean: {active_counts.mean():.2f}, "
        f"median: {np.median(active_counts):.2f}, "
        f"p95: {np.percentile(active_counts, 95):.2f}"
    )

if __name__ == "__main__":
    main()
