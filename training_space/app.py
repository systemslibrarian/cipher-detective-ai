"""
Cipher Detective Training Space
--------------------------------
Trains roberta-base on the classical-cipher-corpus (81 classes, ~58k balanced
examples). Pushes the trained model to
systemslibrarian/cipher-detective-classifier when done.

NO gradio import — uses a bare HTTP server so HF's health check passes
regardless of the audioop/Python-3.13 issue.  Training runs in a background
thread as soon as the process starts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ID   = "systemslibrarian/classical-cipher-corpus"
MODEL_BASE   = "roberta-base"
HUB_MODEL_ID = "systemslibrarian/cipher-detective-classifier"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

EPOCHS       = 10
BATCH_SIZE   = 32
GRAD_ACCUM   = 2
MAX_LEN      = 256
LR           = 2e-5
WARMUP_RATIO = 0.06
LABEL_SMOOTH = 0.05
GAMMA        = 2.0
RESUME_FROM_HUB = True

# ── Shared state ──────────────────────────────────────────────────────────────
_log_lines: list[str] = []
_metrics_history: list[dict] = []
_status = "Initialising…"
_done   = False
_final_metrics: dict = {}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _log_lines.append(line)
    print(line, flush=True)


# ── Focal-loss trainer ────────────────────────────────────────────────────────
def make_focal_trainer(class_weights_tensor, gamma: float = 2.0):
    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            weights = class_weights_tensor.to(logits.device)
            ce      = F.cross_entropy(logits, labels, weight=weights, reduction="none")
            probs   = F.softmax(logits, dim=-1)
            pt      = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal   = ((1 - pt) ** gamma) * ce
            return (focal.mean(), outputs) if return_outputs else focal.mean()
    return FocalTrainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    _metrics_history.append({"accuracy": acc, "macro_f1": f1})
    _log(f"  eval → acc={acc:.4f}  macro_f1={f1:.4f}")
    return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}


# ── Training thread ───────────────────────────────────────────────────────────
def train() -> None:
    global _status, _done, _final_metrics
    try:
        _log("Loading dataset from HF Hub…")
        _status = "Loading dataset…"
        ds = load_dataset(
            DATASET_ID,
            data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"},
            token=HF_TOKEN or None,
        )
        _log(f"  train={len(ds['train']):,}  val={len(ds['validation']):,}")

        labels_sorted = sorted(set(ds["train"]["label"]))
        label2id = {l: i for i, l in enumerate(labels_sorted)}
        id2label  = {i: l for l, i in label2id.items()}
        num_labels = len(labels_sorted)
        _log(f"  {num_labels} labels")

        _log(f"Loading tokenizer: {MODEL_BASE}")
        _status = "Tokenising…"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

        def tokenize(batch):
            return tokenizer(batch["ciphertext"], truncation=True, max_length=MAX_LEN)

        def encode_label(batch):
            batch["labels"] = [label2id[l] for l in batch["label"]]
            return batch

        ds = ds.map(tokenize,     batched=True, batch_size=512, remove_columns=["ciphertext"])
        ds = ds.map(encode_label, batched=True, batch_size=512)
        keep = {"input_ids", "attention_mask", "labels"}
        ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
        ds.set_format("torch")
        _log("Tokenisation complete.")

        from collections import Counter
        counts = Counter(list(ds["train"]["labels"]))
        total  = sum(counts.values())
        weights = np.array(
            [total / (num_labels * counts.get(i, 1)) for i in range(num_labels)],
            dtype=np.float32,
        )
        cw = torch.tensor(weights)
        _log(f"Class weights: min={cw.min():.2f}  max={cw.max():.2f}")

        steps_per_epoch = len(ds["train"]) // (BATCH_SIZE * GRAD_ACCUM)
        eval_steps      = max(50, steps_per_epoch // 2)
        output_dir      = Path("./cipher_model_output")

        # Try to load weights from the latest Hub checkpoint. We deliberately
        # do NOT use trainer.resume_from_checkpoint: the optimizer/scheduler/
        # trainer_state files saved by older Transformers versions are not
        # forward-compatible (missing stateful_callbacks key, parameter-group
        # mismatch, etc.). Loading just the model weights and running with a
        # fresh optimizer is robust across version changes.
        weights_source = MODEL_BASE
        if RESUME_FROM_HUB and HF_TOKEN:
            try:
                from huggingface_hub import snapshot_download
                _log(f"Checking for checkpoint in {HUB_MODEL_ID}…")
                ckpt_dir = Path(
                    snapshot_download(
                        HUB_MODEL_ID,
                        token=HF_TOKEN,
                        ignore_patterns=["*.msgpack", "flax_model*"],
                    )
                )
                ckpts = sorted(
                    ckpt_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[1]),
                )
                if ckpts:
                    candidate = ckpts[-1]
                    # Verify the checkpoint architecture matches MODEL_BASE.
                    # The Hub repo may contain a checkpoint from a *different*
                    # base (e.g. distilbert), in which case loading it with our
                    # roberta tokenizer would produce out-of-range token IDs
                    # and a CUDA device-side assert in the embedding lookup.
                    base_cfg = AutoConfig.from_pretrained(MODEL_BASE)
                    ckpt_cfg = AutoConfig.from_pretrained(str(candidate))
                    if (
                        ckpt_cfg.model_type == base_cfg.model_type
                        and getattr(ckpt_cfg, "vocab_size", None)
                            == getattr(base_cfg, "vocab_size", None)
                    ):
                        weights_source = str(candidate)
                        _log(f"Loading weights from: {candidate.name} (fresh optimizer)")
                    else:
                        _log(
                            f"Checkpoint {candidate.name} is "
                            f"{ckpt_cfg.model_type}/vocab={ckpt_cfg.vocab_size}, "
                            f"but MODEL_BASE is "
                            f"{base_cfg.model_type}/vocab={base_cfg.vocab_size}. "
                            "Ignoring checkpoint and starting from base model."
                        )
                else:
                    _log("No checkpoints found — starting from base model.")
            except Exception as exc:
                _log(f"Could not load checkpoint ({exc}) — starting fresh.")

        _log(f"Loading model from: {weights_source}")
        _status = "Building model…"
        model = AutoModelForSequenceClassification.from_pretrained(
            weights_source,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            label_smoothing_factor=LABEL_SMOOTH,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            report_to="none",
            logging_steps=50,
            push_to_hub=True,
            hub_model_id=HUB_MODEL_ID,
            hub_token=HF_TOKEN,
        )

        FocalTrainer = make_focal_trainer(cw, gamma=GAMMA)
        trainer = FocalTrainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        )

        _status = f"Training ({EPOCHS} epochs, focal loss, A10G)…"
        _log("Starting training…")
        trainer.train()
        _log("Training complete.")

        _status = "Evaluating…"
        result = trainer.evaluate()
        _final_metrics = {k: round(v, 4) for k, v in result.items()}
        _log(f"Final metrics: {_final_metrics}")

        output_dir.mkdir(exist_ok=True)
        (output_dir / "label_mapping.json").write_text(
            json.dumps({"label2id": label2id, "id2label": id2label}, indent=2)
        )

        _status = "Pushing model to Hub…"
        _log(f"Pushing to {HUB_MODEL_ID}…")
        trainer.push_to_hub(
            commit_message=f"trained: macro_f1={_final_metrics.get('eval_macro_f1', '?')}",
            blocking=True,
        )
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(output_dir / "label_mapping.json"),
            path_in_repo="label_mapping.json",
            repo_id=HUB_MODEL_ID,
            commit_message="add label_mapping.json",
        )
        _log(f"Done! https://huggingface.co/{HUB_MODEL_ID}")
        _status = f"Done! macro_f1={_final_metrics.get('eval_macro_f1', '?')}"
        _done = True

    except Exception as exc:
        import traceback
        _log(f"ERROR: {exc}\n{traceback.format_exc()}")
        _status = f"Error: {exc}"
        _done = True


# ── Minimal HTTP server (HF health-check + status page) ──────────────────────
class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = self._build_page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence access log

    def _build_page(self):
        log_html = "\n".join(_log_lines[-60:]).replace("<", "&lt;").replace(">", "&gt;")
        metrics_html = ""
        if _metrics_history:
            rows = "".join(
                f"<tr><td>{i+1}</td><td>{m['accuracy']:.4f}</td><td>{m['macro_f1']:.4f}</td></tr>"
                for i, m in enumerate(_metrics_history)
            )
            metrics_html = f"<h2>Eval history</h2><table border=1><tr><th>Step</th><th>Acc</th><th>macro_F1</th></tr>{rows}</table>"
        final_html = ""
        if _final_metrics:
            items = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in _final_metrics.items())
            final_html = f"<h2>Final metrics</h2><ul>{items}</ul>"
        return f"""<!DOCTYPE html>
<html><head><meta charset=utf-8><meta http-equiv="refresh" content="30">
<title>Cipher Detective Trainer</title>
<style>
body{{font-family:monospace;padding:20px;max-width:960px;background:#0d1117;color:#e6edf3}}
pre{{background:#161b22;color:#3fb950;padding:12px;overflow-x:auto;white-space:pre-wrap;border-radius:6px}}
h1,h2{{color:#58a6ff}}a{{color:#79c0ff}}
table{{border-collapse:collapse;margin:8px 0}}
td,th{{padding:4px 12px;border:1px solid #30363d}}
ul{{line-height:1.8}}
</style>
</head><body>
<h1>&#x1F9E0; Cipher Detective — Training Job</h1>
<p><b>Status:</b> {_status}</p>
<p><b>Model target:</b> <a href="https://huggingface.co/{HUB_MODEL_ID}">{HUB_MODEL_ID}</a></p>
<p><em>Page auto-refreshes every 30 s</em></p>
{metrics_html}
{final_html}
<h2>Training log (last 60 lines)</h2>
<pre>{log_html if log_html else "(starting…)"}</pre>
</body></html>"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.environ.get("TRAIN_ENABLED", "").lower() in ("1", "true", "yes"):
        thread = threading.Thread(target=train, daemon=True)
        thread.start()
    else:
        _status = "Idle — training disabled. Set TRAIN_ENABLED=1 to start."
        _log("Training disabled (TRAIN_ENABLED not set). Serving status page only.")

    # Serve status page on port 7860 (HF Spaces default)
    server = HTTPServer(("0.0.0.0", 7860), StatusHandler)
    _log("Status server running on http://0.0.0.0:7860")
    server.serve_forever()
