from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import IO, BinaryIO

import numpy as np
import regex
import torch
from torch import Tensor, nn
import json

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(2 / (in_features + out_features)), a=-3, b=3)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.t()


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: Tensor, dim: int) -> Tensor:
    z = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(z)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, -1) @ v


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2:
            raise ValueError("RoPE dimension must be even")
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        c, s = self.cos[token_positions], self.sin[token_positions]
        while c.ndim < x.ndim:
            c, s = c.unsqueeze(-3), s.unsqueeze(-3)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None,
                 device=None, dtype=None):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads, self.head_dim, self.rope = num_heads, d_model // num_heads, rope
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        *batch, seq, d_model = x.shape
        def heads(proj: Linear) -> Tensor:
            return proj(x).view(*batch, seq, self.num_heads, self.head_dim).transpose(-3, -2)
        q, k, v = heads(self.q_proj), heads(self.k_proj), heads(self.v_proj)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        causal = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()
        y = scaled_dot_product_attention(q, k, v, causal)
        y = y.transpose(-3, -2).contiguous().view(*batch, seq, d_model)
        return self.output_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 device=None, dtype=None):
        super().__init__()
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device)
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    m = inputs.max(dim=-1).values
    log_z = m + torch.log(torch.exp(inputs - m.unsqueeze(-1)).sum(dim=-1))
    correct = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_z - correct).mean()


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    # Valid starts are [0, len - context_length): a start needs `context_length` inputs
    # plus one more token for the shifted-by-one label.
    starts = np.random.randint(0, len(dataset) - context_length, size=(batch_size, 1))
    ids = starts + np.arange(context_length)[None, :]
    # Index the (possibly memory-mapped) array first so we only materialize the sampled windows.
    x = torch.from_numpy(dataset[ids].astype(np.int64))
    y = torch.from_numpy(dataset[ids + 1].astype(np.int64))
    return x.to(device), y.to(device)


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    norm = torch.sqrt(sum(p.grad.detach().float().square().sum() for p in params))
    scale = min(1.0, max_l2_norm / (norm.item() + 1e-6))
    for p in params:
        p.grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps < 0 or weight_decay < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("invalid AdamW hyperparameter")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = (group[k] for k in ("lr", "betas", "eps", "weight_decay"))
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v, g = state["step"], state["exp_avg"], state["exp_avg_sq"], p.grad
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                alpha = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                p.mul_(1 - lr * wd)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha)
        return loss


def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float,
                           warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        return alpha_max * t / warmup_iters
    if t > cosine_cycle_iters:
        return alpha_min
    phase = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * phase))


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


GPT2_PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def _pretokenize(text: str) -> Iterator[str]:
    """Split text into GPT-2 style pre-tokens; BPE merges never cross these boundaries."""
    return (m.group() for m in GPT2_PATTERN.finditer(text))


def _special_split_re(special_tokens: Iterable[str] | None, capture: bool):
    """Regex that splits on special tokens, longest first so overlapping specials prefer the longer match."""
    tokens = sorted(set(special_tokens or []), key=len, reverse=True)
    if not tokens:
        return None
    alternation = "|".join(map(re.escape, tokens))
    return re.compile(f"({alternation})" if capture else alternation)


# Streaming knobs: small enough that `encode_iterable` holds only a bounded working set.
_STREAM_FLUSH_CHARS = 8192
_CACHE_LIMIT = 8192


def _merge_symbols(symbols: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    result, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = list(special_tokens or [])
        self.bytes_to_id = {token: idx for idx, token in self.vocab.items()}
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self.bytes_to_id:
                self.bytes_to_id[encoded] = len(self.vocab)
                self.vocab[len(self.vocab)] = encoded
        self.special_ids = {s: self.bytes_to_id[s.encode("utf-8")] for s in self.special_tokens}
        self._special_re = _special_split_re(self.special_tokens, capture=True)
        # Streaming holdback: enough tail to never cut a special token or a growing pre-token.
        self._hold = max((len(s) for s in self.special_tokens), default=0) + 1
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json
        with open(vocab_filepath, encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {int(i): bytes.fromhex(v) for i, v in raw.items()}
        merges = []
        with open(merges_filepath, "rb") as f:
            for line in f:
                a, b = line.rstrip(b"\n").split(b" ", 1)
                merges.append((a, b))
        return cls(vocab, merges, special_tokens)

    def _encode_piece(self, piece: bytes) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        symbols = tuple(bytes([b]) for b in piece)
        while len(symbols) > 1:
            # Apply the earliest-learned applicable merge, exactly as during training.
            candidates = ((self.merge_ranks[p], p) for p in zip(symbols, symbols[1:]) if p in self.merge_ranks)
            best = min(candidates, default=None)
            if best is None:
                break
            symbols = _merge_symbols(symbols, best[1])
        ids = tuple(self.bytes_to_id[s] for s in symbols)
        # Bounded so that streaming a large corpus stays within a fixed memory budget.
        if len(self._cache) >= _CACHE_LIMIT:
            self._cache.clear()
        self._cache[piece] = ids
        return ids

    def _split_specials(self, text: str) -> list[str]:
        """Alternating [text, special, text, ...] parts; even indices are ordinary text."""
        return self._special_re.split(text) if self._special_re is not None else [text]

    def _encode_stream(self, text: str) -> Iterator[int]:
        for index, part in enumerate(self._split_specials(text)):
            if not part:
                continue
            if index % 2:
                yield self.special_ids[part]
            else:
                for pretoken in _pretokenize(part):
                    yield from self._encode_piece(pretoken.encode("utf-8"))

    def encode(self, text: str) -> list[int]:
        return list(self._encode_stream(text))

    def _flush_boundary(self, buffer: str) -> int:
        """
        Longest prefix of `buffer` that tokenizes the same in isolation as it would in the
        full stream. Anything within `_hold` chars of the end may still be a partial special
        token, and the trailing pre-token may still grow, so both stay buffered.
        """
        limit = len(buffer) - self._hold
        if limit <= 0:
            return 0
        cut = position = 0
        for index, part in enumerate(self._split_specials(buffer)):
            start, position = position, position + len(part)
            if index % 2:
                # A special token is only settled once it sits fully inside the safe region;
                # otherwise a longer overlapping special could still be forming.
                if position > limit:
                    break
                cut = position
                continue
            # Pre-tokens tile the segment, so its last safe boundary is the last match end <= limit.
            for match in GPT2_PATTERN.finditer(part):
                if start + match.end() > limit:
                    break
                cut = start + match.end()
            if position > limit:
                break
        return cut

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if len(buffer) < _STREAM_FLUSH_CHARS:
                continue
            cut = self._flush_boundary(buffer)
            if cut:
                yield from self._encode_stream(buffer[:cut])
                buffer = buffer[cut:]
        yield from self._encode_stream(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs):
    if vocab_size < len(special_tokens) + 256:
        raise ValueError("vocab_size must leave room for the 256 byte tokens and the special tokens")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # Special tokens are removed before pre-tokenization, so no merge can ever span
    # a document boundary or produce a token containing a special-token fragment.
    splitter = _special_split_re(special_tokens, capture=False)
    documents = splitter.split(text) if splitter is not None else [text]

    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(_pretokenize(document))

    # Parallel arrays: the symbol sequence of each distinct pre-token, and how often it occurs.
    words = [tuple(bytes([b]) for b in word.encode("utf-8")) for word in frequencies]
    counts = list(frequencies.values())

    # Incremental bookkeeping: total frequency per adjacent pair, and which words contain it.
    # Rebuilding these from scratch on every merge is what makes the naive trainer quadratic.
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for i, symbols in enumerate(words):
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += counts[i]
            pair_words[pair].add(i)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and pair_counts:
        # Most frequent pair, breaking ties toward the lexicographically greatest pair.
        best = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        for i in list(pair_words[best]):
            symbols, count = words[i], counts[i]
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_words[pair].discard(i)
            symbols = _merge_symbols(symbols, best)
            words[i] = symbols
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += count
                pair_words[pair].add(i)
        pair_words.pop(best, None)

    return vocab, merges

def load_bpe(
    path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    vocab = {
        int(token_id): bytes.fromhex(token_hex)
        for token_id, token_hex in data["vocab"].items()
    }

    merges = [
        (bytes.fromhex(left), bytes.fromhex(right))
        for left, right in data["merges"]
    ]

    return vocab, merges

def save_bpe(
    path: str | os.PathLike,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> None:
    data = {
        "vocab": {
            str(token_id): token_bytes.hex()
            for token_id, token_bytes in vocab.items()
        },
        "merges": [
            [left.hex(), right.hex()]
            for left, right in merges
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    