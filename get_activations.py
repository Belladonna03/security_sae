import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen3-4B"
CSV_PATH = "/Users/dekovaleva/PythonProjects/security_sae/dataset/dataset.csv"
SAE_CKPT_PATH = "/Users/dekovaleva/PythonProjects/security_sae/activations/ckpt_epoch_15.pt"

TEXT_COL = "prompt"
LAYER_IDX = 20
MAX_LENGTH = 512
BATCH_SIZE = 2
NUM_WORKERS = 0
TOKEN_SUBSAMPLE_P = 1.0

OUT_DIR = "./sae_feats_layer20"
SAVE_EVERY_N_BATCHES = 50

LLM_DTYPE = torch.bfloat16
SAE_DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_layers(model):
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Не нашёл model.model")
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "h"):
        return base.h
    raise RuntimeError("Не нашёл список слоёв")


class ActivationExtractor:
    """Снимает активации одного слоя (выход блока) и возвращает [N,D] без padding."""
    def __init__(self, model, layer_idx: int, token_subsample_p: float = 1.0, seed: int = 0):
        self.model = model
        self.layers = get_layers(model)
        if not (0 <= layer_idx < len(self.layers)):
            raise ValueError(
                f"Invalid layer_idx={layer_idx}, num_layers={len(self.layers)}"
            )
        self.layer_idx = layer_idx
        self.p = float(token_subsample_p)
        self.rng = np.random.default_rng(seed)
        self._captured = None
        self._handle = self.layers[self.layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self._captured = output

    def close(self):
        self._handle.remove()

    @property
    def d_model(self) -> int:
        return int(getattr(self.model.config, "hidden_size"))

    def extract_from_batch(self, batch: dict) -> torch.Tensor:
        dev = self.model.get_input_embeddings().weight.device
        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)

        self._captured = None
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        acts = self._captured
        if acts is None:
            raise RuntimeError("Hook did not capture activations")

        valid = attention_mask.bool()
        acts_valid = acts[valid]

        if acts_valid.numel() == 0:
            return acts_valid

        if self.p < 1.0:
            n = acts_valid.shape[0]
            keep = self.rng.random(n) < self.p
            if not np.any(keep):
                return acts_valid[:0]
            keep_t = torch.from_numpy(keep).to(acts_valid.device)
            acts_valid = acts_valid[keep_t]

        return acts_valid


class JumpReLUSAE(nn.Module):
    def __init__(self, d_in, d_feat):
        super().__init__()
        self.d_in = d_in
        self.d_feat = d_feat

        self.encoder = nn.Linear(d_in, d_feat, bias=True)
        self.decoder = nn.Linear(d_feat, d_in, bias=True)
        self.theta = nn.Parameter(torch.zeros(d_feat))

        nn.init.normal_(self.encoder.weight, std=0.02)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        z = self.encoder(x)
        a = F.relu(z - self.theta)
        return a

    def forward(self, x):
        a = self.encode(x)
        x_hat = self.decoder(a)
        return x_hat, a


def load_sae_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise ValueError(
            f"Expected SAE checkpoint with 'model' key. Got keys: {list(ckpt.keys())}"
        )
    if not isinstance(ckpt["model"], dict):
        raise ValueError(f"Expected 'model' to be dict, got {type(ckpt['model'])}")
    return ckpt


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, text_col: str):
        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            raise ValueError(f"TEXT_COL='{text_col}' not found. Columns: {list(df.columns)}")
        self.texts = df[text_col].astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}


def make_collate_fn(tokenizer, max_length):
    def collate(examples):
        texts = [ex["text"] for ex in examples]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    return collate


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LLM
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=LLM_DTYPE,
        device_map="auto",
    )
    llm.eval()

    extractor = ActivationExtractor(
        llm, layer_idx=LAYER_IDX, token_subsample_p=TOKEN_SUBSAMPLE_P, seed=0
    )
    num_layers = len(get_layers(llm))
    d_model = extractor.d_model
    print(f"[llm] num_layers={num_layers}, d_model={d_model}, layer_idx={LAYER_IDX}")

    D_IN = 2560
    D_FEAT = 20480

    if d_model != D_IN:
        raise RuntimeError(
            f"LLM d_model={d_model} != SAE d_in={D_IN}. "
            f"SAE was trained for different model/layer. "
            f"Check MODEL_ID/LAYER_IDX or retrain SAE for d_model={d_model}."
        )

    ckpt = load_sae_ckpt(SAE_CKPT_PATH)
    sae = JumpReLUSAE(d_in=D_IN, d_feat=D_FEAT).to(device=DEVICE, dtype=SAE_DTYPE)
    sae.load_state_dict(ckpt["model"], strict=True)
    sae.eval()
    print(f"[sae] loaded. step={ckpt.get('step', None)}")

    ds = CSVDataset(CSV_PATH, TEXT_COL)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=make_collate_fn(tokenizer, MAX_LENGTH),
        pin_memory=True,
    )

    meta = {
        "MODEL_ID": MODEL_ID,
        "CSV_PATH": CSV_PATH,
        "SAE_CKPT_PATH": SAE_CKPT_PATH,
        "TEXT_COL": TEXT_COL,
        "MAX_LENGTH": MAX_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "LAYER_IDX": LAYER_IDX,
        "NUM_LAYERS": num_layers,
        "D_MODEL": d_model,
        "SAE_D_IN": D_IN,
        "SAE_D_FEAT": D_FEAT,
        "TOKEN_SUBSAMPLE_P": TOKEN_SUBSAMPLE_P,
        "LLM_DTYPE": str(LLM_DTYPE),
        "SAE_DTYPE": str(SAE_DTYPE),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_tokens = 0
    chunk_idx = 0
    feats_buf = []

    def flush():
        nonlocal chunk_idx, feats_buf
        if not feats_buf:
            return
        feats = torch.cat(feats_buf, dim=0)
        assert len(feats.shape) == 2 and feats.shape[1] == D_FEAT, \
            f"Expected shape [N, {D_FEAT}], got {feats.shape}"
        out_path = os.path.join(
            OUT_DIR, f"sae_feats_layer{LAYER_IDX:02d}_chunk{chunk_idx:05d}.pt"
        )
        torch.save(
            {"sae_feats": feats, "total_tokens_so_far": total_tokens}, out_path
        )
        print(f"[saved] {out_path} shape={tuple(feats.shape)}")
        feats_buf = []
        chunk_idx += 1

    try:
        for i, batch in enumerate(loader):
            llm_acts = extractor.extract_from_batch(batch)
            if llm_acts.numel() == 0:
                continue
            total_tokens += int(llm_acts.shape[0])

            with torch.no_grad():
                a = sae.encode(llm_acts.to(device=DEVICE, dtype=SAE_DTYPE))

            feats_buf.append(a.detach().to("cpu"))

            if (i + 1) % SAVE_EVERY_N_BATCHES == 0:
                flush()
                print(f"[progress] batch={i+1}/{len(loader)} total_tokens={total_tokens}")

        flush()
        print(f"[done] total_tokens={total_tokens}")

    finally:
        extractor.close()


if __name__ == "__main__":
    main()
