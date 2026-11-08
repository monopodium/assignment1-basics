#!/usr/bin/env bash
# Train a small Transformer LM on the full TinyStories corpus, on one GPU.
#
#   bash tests/run_tinystories.sh                        # data/TinyStoriesV2-GPT4-{train,valid}.txt
#   WANDB_PROJECT=cs336-a1 bash tests/run_tinystories.sh # also stream metrics to Weights & Biases
#
# Every knob below can be overridden from the environment, e.g.
#   MAX_ITERS=200 BATCH_SIZE=16 bash tests/run_tinystories.sh
#
# Smoke test on the 5MB fixture (VALID_CORPUS empty => hold out VALID_FRACTION of it):
#   CORPUS=tests/fixtures/tinystories_sample_5M.txt VALID_CORPUS= OUT_DIR=runs/smoke \
#     MAX_ITERS=60 bash tests/run_tinystories.sh
set -euo pipefail
export WANDB_API_KEY=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
cd "$(dirname "$0")/.."

WANDB_PROJECT=${WANDB_PROJECT-cs336-a1}   # WANDB_PROJECT= (empty) turns W&B off
CORPUS=${CORPUS:-data/TinyStoriesV2-GPT4-train.txt}
# Official held-out split. Point VALID_CORPUS at nothing to carve VALID_FRACTION of the
# documents out of $CORPUS instead (what the 5MB fixture needs, since it ships no valid set).
VALID_CORPUS=${VALID_CORPUS-data/TinyStoriesV2-GPT4-valid.txt}
OUT_DIR=${OUT_DIR:-runs/tinystories-full}
RUN_NAME=${RUN_NAME:-tinystories-full-4L-512d}
VALID_FRACTION=${VALID_FRACTION:-0.02}   # share of documents held out for validation

# Tokenizer artifact, deliberately outside $OUT_DIR so every run shares one BPE.
# Trained on first use and reused forever after; delete the file to retrain.
BPE_JSON=${BPE_JSON:-data/bpe-10000.json}
VOCAB_SIZE=${VOCAB_SIZE:-10000}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256}
D_MODEL=${D_MODEL:-512}
D_FF=${D_FF:-1344}
NUM_LAYERS=${NUM_LAYERS:-4}
NUM_HEADS=${NUM_HEADS:-16}
ROPE_THETA=${ROPE_THETA:-10000}

LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
WARMUP_ITERS=${WARMUP_ITERS:-200}
# The assignment's token budget: BATCH_SIZE * MAX_ITERS * CONTEXT_LENGTH = 327,680,000.
# Trade batch size against step count freely, but keep that product fixed.
BATCH_SIZE=${BATCH_SIZE:-256}
MAX_ITERS=${MAX_ITERS:-5000}

EVAL_INTERVAL=${EVAL_INTERVAL:-250}
EVAL_ITERS=${EVAL_ITERS:-20}
LOG_INTERVAL=${LOG_INTERVAL:-25}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1000}

DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # single GPU; this script is not distributed

mkdir -p "$OUT_DIR"

echo "[budget] $BATCH_SIZE (batch) x $MAX_ITERS (steps) x $CONTEXT_LENGTH (context) = \
$((BATCH_SIZE * MAX_ITERS * CONTEXT_LENGTH)) tokens processed"

if [[ ! -s "$CORPUS" ]]; then
  echo "missing corpus $CORPUS -- download it with the wget commands in README.md" >&2
  exit 1
fi

if [[ -n "$VALID_CORPUS" ]]; then
  # Official train/valid files: feed them straight to the trainer, no multi-GB copy.
  TRAIN_TXT="$CORPUS"
  VALID_TXT="$VALID_CORPUS"
  echo "[split] using the official split: $TRAIN_TXT + $VALID_TXT"
else
# --- split the corpus on document boundaries (done once, then cached) -------------------
TRAIN_TXT="$OUT_DIR/train.txt"
VALID_TXT="$OUT_DIR/valid.txt"
if [[ ! -s "$TRAIN_TXT" || ! -s "$VALID_TXT" ]]; then
  echo "[split] $CORPUS -> $TRAIN_TXT + $VALID_TXT (valid=$VALID_FRACTION)"
  CORPUS="$CORPUS" TRAIN_TXT="$TRAIN_TXT" VALID_TXT="$VALID_TXT" VALID_FRACTION="$VALID_FRACTION" \
  python - <<'PY'
import os
sep = "<|endoftext|>"
text = open(os.environ["CORPUS"], encoding="utf-8").read()
# Split between documents so no story is cut in half across the train/valid boundary.
docs = [d for d in text.split(sep) if d.strip()]
cut = max(1, int(len(docs) * (1 - float(os.environ["VALID_FRACTION"]))))
for path, part in ((os.environ["TRAIN_TXT"], docs[:cut]), (os.environ["VALID_TXT"], docs[cut:])):
    with open(path, "w", encoding="utf-8") as f:
        f.write(sep.join(part) + sep)
    print(f"  {path}: {len(part)} docs, {os.path.getsize(path) / 1e6:.1f} MB")
PY
else
  echo "[split] reusing $TRAIN_TXT + $VALID_TXT"
fi
fi

# --- optional Weights & Biases ----------------------------------------------------------
# Unset WANDB_PROJECT and the run logs to the console and $OUT_DIR/log.jsonl only.
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_ARGS=(--wandb-project "$WANDB_PROJECT")
  echo "[wandb] project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
fi

# The first run over a corpus encodes both splits into $OUT_DIR/*.bin (~13 min for the
# full 2.2GB set); later runs reuse $BPE_JSON and those .bin files instantly.
set -x
python -u tests/train_script.py \
  --input-path "$TRAIN_TXT" \
  --valid-path "$VALID_TXT" \
  --output-path "$OUT_DIR" \
  --vocab-path "$BPE_JSON" \
  --vocab-size "$VOCAB_SIZE" \
  --special-tokens '<|endoftext|>' \
  --context-length "$CONTEXT_LENGTH" \
  --d-model "$D_MODEL" \
  --d-ff "$D_FF" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --rope-theta "$ROPE_THETA" \
  --lr "$LR" \
  --min-lr "$MIN_LR" \
  --warmup-iters "$WARMUP_ITERS" \
  --batch-size "$BATCH_SIZE" \
  --max-iters "$MAX_ITERS" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-iters "$EVAL_ITERS" \
  --log-interval "$LOG_INTERVAL" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --run-name "$RUN_NAME" \
  "${WANDB_ARGS[@]}"
