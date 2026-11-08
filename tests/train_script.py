"""Command-line entry point for training a Transformer LM.

Usage:
    uv run python tests/train_script.py --input-path data/train.npy --output-path runs/tinystories
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cs336_basics.transformer import (
    AdamW,
    Embedding,
    Linear,
    MultiheadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    Tokenizer,
    TransformerBlock,
    TransformerLM,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_bpe,
    load_checkpoint,
    save_bpe,
    save_checkpoint,
    scaled_dot_product_attention,
    silu,
    softmax,
    train_bpe,
)

TOKEN_ARRAY_SUFFIXES = {".npy", ".bin", ".dat"}
ENCODE_FLUSH = 1 << 20  # ids buffered before being appended to the cache file

        
def decode(model:TransformerLM,tokenizer:Tokenizer,request:str, temperature:float, p: float,max_new_tokens: int = 100, end_token_id: int | None = None,) -> str:
    model.eval()
    device = next(model.parameters()).device
    tokens = torch.Tensor(tokenizer.encode(request),dtype=torch.long, device = device).unsqueeze(0)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            next_logits = model(tokens)[0,-1]
            if temperature == 0:
                next_token = torch.argmax(next_logits)
            else:
                next_logits = next_logits/temperature
                probalilities = softmax(next_logits, dim = -1)
                sorted_probs, sorted_indices = torch.sort(probalilities,descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                keep = cumulative_probs - sorted_probs < p
                filtered_probs = sorted_probs * keep
                filtered_probs = (filtered_probs/filtered_probs.sum())
                sampled_position = torch.multinomial(filtered_probs,num_samples = 1)
                next_token = sorted_indices[sampled_position].squeeze(0)
            tokens = torch.cat([tokens,next_token.view(1,1)],dim=1)
            if end_token_id is not None and next_token.item() == end_token_id:
                break
    if was_training:
        model.train()
    
    return tokenizer.decode(tokens[0].tolist())
    
    
    
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--input-path", type=Path, required=True,
                      help="Tokenized training set (1D uint16/int32 .npy, loaded with mmap).")
    data.add_argument("--valid-path", type=Path, default=None,
                      help="Tokenized validation set; skips evaluation when omitted.")
    data.add_argument("--output-path", type=Path, required=True,
                      help="Directory for checkpoints, logs and the final model.")

    tokenizer = parser.add_argument_group("tokenizer")
    tokenizer.add_argument("--vocab-size", type=int, default=10_000)
    tokenizer.add_argument("--vocab-path", type=Path, default=None)
    tokenizer.add_argument("--merges-path", type=Path, default=None)
    tokenizer.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])

    model = parser.add_argument_group("model")
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    optim = parser.add_argument_group("optimizer")
    optim.add_argument("--lr", type=float, default=3e-4, help="Peak (post-warmup) learning rate.")
    optim.add_argument("--min-lr", type=float, default=3e-5, help="Final learning rate of the cosine cycle.")
    optim.add_argument("--warmup-iters", type=int, default=200)
    optim.add_argument("--cosine-cycle-iters", type=int, default=None,
                       help="Cosine cycle length; defaults to --max-iters.")
    optim.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), metavar=("BETA1", "BETA2"))
    optim.add_argument("--eps", type=float, default=1e-8)
    optim.add_argument("--weight-decay", type=float, default=0.1)
    optim.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Gradient clipping threshold; 0 disables clipping.")

    loop = parser.add_argument_group("training loop")
    loop.add_argument("--batch-size", type=int, default=32)
    loop.add_argument("--max-iters", type=int, default=5_000)
    loop.add_argument("--eval-interval", type=int, default=500)
    loop.add_argument("--eval-iters", type=int, default=50, help="Batches averaged per evaluation.")
    loop.add_argument("--log-interval", type=int, default=10)

    ckpt = parser.add_argument_group("checkpointing")
    ckpt.add_argument("--checkpoint-interval", type=int, default=1_000)
    ckpt.add_argument("--resume-from", type=Path, default=None,
                      help="Checkpoint to restore model, optimizer and iteration count from.")

    system = parser.add_argument_group("system")
    system.add_argument("--device", type=str, default="cuda")
    system.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    system.add_argument("--seed", type=int, default=42)
    system.add_argument("--compile", action="store_true", help="Wrap the model in torch.compile.")

    logging = parser.add_argument_group("logging")
    logging.add_argument("--wandb-project", type=str, default=None, help="Disables W&B when omitted.")
    logging.add_argument("--run-name", type=str, default=None)

    return parser


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def token_dtype(vocab_size: int) -> np.dtype:
    """Smallest integer type that can hold every id of this vocabulary."""
    return np.dtype(np.uint16 if vocab_size < 2**16 else np.int32)


def build_tokenizer(args: argparse.Namespace) -> Tokenizer:
    """Load a BPE tokenizer from disk, or train one on --input-path and cache it."""
    specials = list(args.special_tokens or ["<|endoftext|>"])

    # (a) the assignment's two-file format: hex vocab json + merges txt
    if args.vocab_path is not None and args.merges_path is not None \
            and args.vocab_path.exists() and args.merges_path.exists():
        print(f"[tokenizer] loading {args.vocab_path} + {args.merges_path}")
        return Tokenizer.from_files(args.vocab_path, args.merges_path, specials)

    # (b) a single json written by save_bpe
    cache = args.vocab_path if args.vocab_path is not None else args.output_path / "bpe.json"
    if cache.exists():
        print(f"[tokenizer] loading {cache}")
        vocab, merges = load_bpe(cache)
        return Tokenizer(vocab, merges, specials)

    # (c) nothing on disk: train from scratch on the raw training text
    print(f"[tokenizer] training BPE (vocab_size={args.vocab_size}) on {args.input_path} ...")
    start = time.perf_counter()
    vocab, merges = train_bpe(args.input_path, vocab_size=args.vocab_size, special_tokens=specials)
    print(f"[tokenizer] trained {len(vocab)} tokens / {len(merges)} merges "
          f"in {time.perf_counter() - start:.1f}s -> {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_bpe(cache, vocab, merges)
    return Tokenizer(vocab, merges, specials)


def encode_to_cache(text_path: Path, tokenizer: Tokenizer, cache_path: Path, dtype: np.dtype) -> None:
    """Stream-encode a text file into a flat binary array of token ids.

    `encode_iterable` keeps memory bounded on the read side; flushing every ENCODE_FLUSH
    ids keeps it bounded on the write side, so arbitrarily large corpora are fine.
    """
    print(f"[data] encoding {text_path} -> {cache_path}")
    start, total = time.perf_counter(), 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".partial")
    buffer: list[int] = []
    with open(text_path, encoding="utf-8") as src, open(tmp, "wb") as dst:
        for token_id in tokenizer.encode_iterable(src):
            buffer.append(token_id)
            if len(buffer) >= ENCODE_FLUSH:
                np.asarray(buffer, dtype=dtype).tofile(dst)
                total += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(dst)
            total += len(buffer)
    tmp.replace(cache_path)  # only rename once complete, so a crash never leaves a valid-looking cache
    print(f"[data] {total:,} tokens in {time.perf_counter() - start:.1f}s")


def load_tokens(path: Path, args: argparse.Namespace, tokenizer_factory) -> np.ndarray:
    """Memory-map `path`, tokenizing it first (with caching) when it is raw text."""
    dtype = token_dtype(args.vocab_size)

    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix in TOKEN_ARRAY_SUFFIXES:
        return np.memmap(path, dtype=dtype, mode="r")

    # Raw text: encode once into <output>/<stem>-<vocab_size>.bin and reuse it afterwards.
    cache = args.output_path / f"{path.stem}-{args.vocab_size}-{dtype.name}.bin"
    if not cache.exists():
        encode_to_cache(path, tokenizer_factory(), cache, dtype)
    return np.memmap(cache, dtype=dtype, mode="r")


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------

def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[system] CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate(model, data: np.ndarray, args: argparse.Namespace, device, autocast) -> float:
    """Mean cross-entropy over `--eval-iters` freshly sampled batches."""
    model.eval()
    total = 0.0
    for _ in range(args.eval_iters):
        x, y = get_batch(data, args.batch_size, args.context_length, device)
        with autocast():
            loss = cross_entropy(model(x), y)
        total += loss.item()
    model.train()
    return total / max(args.eval_iters, 1)


def train(args: argparse.Namespace) -> None:
    """Run the full training loop for the configuration in `args`."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    args.output_path.mkdir(parents=True, exist_ok=True)
    (args.output_path / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
    )

    # Mixed precision through autocast: parameters and the optimizer stay in fp32, only the
    # forward pass runs in the low-precision dtype, which is what makes bf16/fp16 stable here.
    amp_dtype = getattr(torch, args.dtype)
    use_amp = amp_dtype is not torch.float32 and device.type == "cuda"
    autocast = (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    # --- data: one tokenizer at most, built lazily and shared by both splits ------------
    cached_tokenizer: list[Tokenizer] = []

    def tokenizer_factory() -> Tokenizer:
        if not cached_tokenizer:
            cached_tokenizer.append(build_tokenizer(args))
        return cached_tokenizer[0]

    train_data = load_tokens(args.input_path, args, tokenizer_factory)
    valid_data = load_tokens(args.valid_path, args, tokenizer_factory) if args.valid_path else None
    print(f"[data] train={len(train_data):,} tokens"
          + (f" valid={len(valid_data):,} tokens" if valid_data is not None else " (no validation set)"))
    if len(train_data) <= args.context_length:
        raise ValueError(f"training set has {len(train_data)} tokens, needs > context_length "
                         f"({args.context_length}) to sample a single batch")

    # --- model / optimizer --------------------------------------------------------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {num_params / 1e6:.1f}M parameters on {device}")

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=tuple(args.betas),
                      eps=args.eps, weight_decay=args.weight_decay)

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"[checkpoint] resumed from {args.resume_from} at iteration {start_iter}")

    # Compile after loading so the checkpoint's state_dict keys match the plain module.
    step_model = torch.compile(model) if args.compile else model

    # --- logging ------------------------------------------------------------------------
    run_name = args.run_name or args.output_path.name
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        except Exception as exc:  # a missing/unauthenticated W&B must not abort training
            print(f"[wandb] disabled ({exc})")
    log_file = (args.output_path / "log.jsonl").open("a")

    def record(**fields) -> None:
        log_file.write(json.dumps(fields) + "\n")
        log_file.flush()
        if wandb_run is not None:
            wandb_run.log(fields, step=fields["iter"])

    def checkpoint(iteration: int, name: str | None = None) -> Path:
        path = args.output_path / (name or f"ckpt_{iteration:06d}.pt")
        save_checkpoint(model, optimizer, iteration, path)
        print(f"[checkpoint] wrote {path}")
        return path

    # --- loop ---------------------------------------------------------------------------
    tokens_per_iter = args.batch_size * args.context_length
    model.train()
    last_log, running = time.perf_counter(), 0.0
    iteration = start_iter

    try:
        for iteration in range(start_iter, args.max_iters):
            lr = get_lr_cosine_schedule(iteration, args.lr, args.min_lr,
                                        args.warmup_iters, args.cosine_cycle_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            with autocast():
                loss = cross_entropy(step_model(x), y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                gradient_clipping(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

            if args.log_interval and (iteration + 1) % args.log_interval == 0:
                now = time.perf_counter()
                elapsed = now - last_log
                mean_loss = running / args.log_interval
                record(iter=iteration + 1, train_loss=mean_loss, lr=lr,
                       ms_per_iter=1e3 * elapsed / args.log_interval,
                       tokens_per_sec=tokens_per_iter * args.log_interval / elapsed,
                       wallclock=now)
                print(f"iter {iteration + 1:>7}/{args.max_iters}  loss {mean_loss:.4f}  "
                      f"lr {lr:.3e}  {1e3 * elapsed / args.log_interval:6.1f} ms/it")
                last_log, running = time.perf_counter(), 0.0

            if valid_data is not None and args.eval_interval and (iteration + 1) % args.eval_interval == 0:
                val_loss = evaluate(step_model, valid_data, args, device, autocast)
                record(iter=iteration + 1, valid_loss=val_loss)
                print(f"iter {iteration + 1:>7}/{args.max_iters}  valid loss {val_loss:.4f}")
                last_log = time.perf_counter()  # don't bill evaluation time to the next window

            if args.checkpoint_interval and (iteration + 1) % args.checkpoint_interval == 0:
                checkpoint(iteration + 1)
    except KeyboardInterrupt:
        print("\n[train] interrupted, saving ...")
        checkpoint(iteration, name="ckpt_interrupt.pt")
        raise
    finally:
        log_file.close()

    if valid_data is not None:
        final_val = evaluate(step_model, valid_data, args, device, autocast)
        print(f"[train] final valid loss {final_val:.4f}")
    checkpoint(args.max_iters, name="ckpt_final.pt")
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = build_parser().parse_args()
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters
    train(args)


if __name__ == "__main__":
    main()
