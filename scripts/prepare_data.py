"""Baixa (streaming), mistura e tokeniza o corpus com o tokenizer do modelo base.

Uso:
    uv run python scripts/prepare_data.py configs/data.yaml

Gera data/train.bin, data/val.bin (uint32) e data/manifest.json com origem, licença,
contagem de tokens e fertilidade (tokens por palavra) de cada fonte.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forja_pretrain.data import TOKEN_DTYPE  # noqa: E402


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    tok = Tokenizer.from_pretrained(cfg["tokenizer"])
    eos = tok.token_to_id(cfg["eos_token"])
    if eos is None:
        raise ValueError(f"eos_token {cfg['eos_token']!r} não existe no tokenizer")

    out = Path(cfg.get("out_dir", "data"))
    out.mkdir(parents=True, exist_ok=True)
    total_budget = int(cfg["total_tokens"])
    val_frac = float(cfg.get("val_fraction", 0.01))

    train_f = (out / "train.bin").open("wb")
    val_f = (out / "val.bin").open("wb")
    seen_hashes: set[bytes] = set()
    manifest = {"tokenizer": cfg["tokenizer"], "sources": []}

    for src in cfg["sources"]:
        budget = int(total_budget * src["weight"])
        ds = load_dataset(src["path"], src.get("name"), split=src.get("split", "train"), streaming=True)
        n_tok = n_docs = n_words = n_dups = 0
        batch: list[str] = []

        def flush(batch: list[str]) -> int:
            nonlocal n_tok, n_docs, n_words
            written = 0
            for enc, text in zip(tok.encode_batch(batch, add_special_tokens=False), batch, strict=True):
                ids = np.array([*enc.ids, eos], dtype=TOKEN_DTYPE)
                # Split por documento (hash), não por token: evita vazar o mesmo doc em treino e validação.
                h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                is_val = int.from_bytes(h, "little") / 2**64 < val_frac
                ids.tofile(val_f if is_val else train_f)
                n_tok += len(ids)
                n_docs += 1
                n_words += len(text.split())
                written += len(ids)
            return written

        for row in ds:
            text = row[src["text_field"]]
            if not text or len(text) < src.get("min_chars", 200):
                continue
            h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
            if h in seen_hashes:  # dedup exato; near-dup (MinHash) fica para v2
                n_dups += 1
                continue
            seen_hashes.add(h)
            batch.append(text)
            if len(batch) == 256:
                flush(batch)
                batch = []
                print(f"\r[{src['path']}] {n_tok / 1e6:.1f}M / {budget / 1e6:.1f}M tokens", end="")
                if n_tok >= budget:
                    break
        if batch and n_tok < budget:
            flush(batch)
        print()
        manifest["sources"].append(
            {
                **{k: src.get(k) for k in ("path", "name", "split", "license", "weight")},
                "tokens": n_tok,
                "docs": n_docs,
                "exact_dups_removed": n_dups,
                "fertility_tokens_per_word": round(n_tok / max(1, n_words), 3),
            }
        )

    train_f.close()
    val_f.close()
    for split in ("train", "val"):
        manifest[f"{split}_tokens"] = (out / f"{split}.bin").stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/data.yaml")
