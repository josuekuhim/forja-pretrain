"""Dados tokenizados em disco (uint32, memmap) e amostragem de janelas aleatórias.

uint32 porque o vocab do Qwen (~152k) não cabe em uint16.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint32


class TokenStream:
    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(f"{self.path} tem {len(self.tokens)} tokens, menos que seq_len+1")

    def __len__(self) -> int:
        return len(self.tokens)

    def sample(self, batch_size: int, generator: torch.Generator, device: str | torch.device = "cpu"):
        hi = len(self.tokens) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
        buf = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(buf)
        x, y = t[:, :-1], t[:, 1:]
        if torch.device(device).type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def write_tokens(path: str | Path, ids: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=TOKEN_DTYPE).tofile(path)
