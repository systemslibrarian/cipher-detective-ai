"""
Cipher Detective Training Space
--------------------------------
Trains roberta-base on the classical-cipher-corpus (81 classes, ~58k examples).
Pushes the trained model to systemslibrarian/cipher-detective-classifier.

Training starts automatically on Space startup. The Gradio UI shows
live progress and final metrics.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
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
GRAD_ACCUM   = 2          # effective batch = 64
MAX_LEN      = 256
LR           = 2e-5
WARMUP_RATIO = 0.06
LABEL_SMOOTH = 0.05
GAMMA        = 2.0        # focal loss gamma
RESUME_FROM_HUB = True   # resume from latest checkpoint in HUB_MODEL_ID if present

# ── Shared state ──────────────────────────────────────────────────────────────
_log_lines: list[str] = []
_metrics_history: list[dict] = []
_status = "⏳ Initialising…"
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
        _status = "📦 Loading dataset…"
        ds = load_dataset(DATASET_ID, data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})
        _log(f"  train={len(ds['train']):,}  val={len(ds['validation']):,}")

        labels_sorted = sorted(set(ds["train"]["label"]))
        label2id = {l: i for i, l in enumerate(labels_sorted)}
        id2label  = {i: l for l, i in label2id.items()}
        num_labels = len(labels_sorted)
        _log(f"  {num_labels} labels")

        _log(f"Loading tokenizer: {MODEL_BASE}")
        _status = "🔤 Tokenising…"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

        def tokenize(batch):
            return tokenizer(batch["ciphertext"], truncation=True, max_length=MAX_LEN)

        def encode_label(batch):
            batch["labels"] = [label2id[l] for l in batch["label"]]
            return batch

        ds = ds.map(tokenize,      batched=True, batch_size=512, remove_columns=["ciphertext"])
        ds = ds.map(encode_label,  batched=True, batch_size=512)
        ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("input_ids", "attention_mask", "labels")])
        ds.set_format("torch")
        _log("Tokenisation complete.")

        # Class weights for imbalanced minority classes
        from collections import Counter
        counts = Counter(ds["train"]["labels"].tolist())
        total  = sum(counts.values())
        weights = np.array([total / (num_labels * counts.get(i, 1)) for i in range(num_labels)], dtype=np.float32)
        cw = torch.tensor(weights)
        _log(f"Class weights: min={cw.min():.2f}  max={cw.max():.2f}")

        _log(f"Loading model: {MODEL_BASE} ({num_labels} output labels)")
        _status = "🏗️ Building model…"
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_BASE,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        steps_per_epoch = len(ds["train"]) // (BATCH_SIZE * GRAD_ACCUM)
        total_steps     = steps_per_epoch * EPOCHS
        eval_steps      = max(50, steps_per_epoch // 2)

        output_dir = Path("./cipher_model_output")

        # Resume from latest Hub checkpoint if available
        resume_checkpoint = None
        if RESUME_FROM_HUB and HF_TOKEN:
            try:
                from huggingface_hub import snapshot_download
                _log(f"Checking for existing checkpoint in {HUB_MODEL_ID}…")
                ckpt_dir = Path(snapshot_download(HUB_MODEL_ID, token=HF_TOKEN, ignore_patterns=["*.msgpack", "flax_model*"]))
                # Find highest numbered checkpoint
                ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
                if ckpts:
                    resume_checkpoint = str(ckpts[-1])
                    _log(f"Resuming from: {ckpts[-1].name}")
                else:
                    _log("No checkpoints found — starting fresh.")
            except Exception as e:
                _log(f"Could not load checkpoint ({e}) — starting fresh.")
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            label_smoothing_factor=LABEL_SMOOTH,
            evaluation_strategy="steps",
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
            logging_dir=str(output_dir / "logs"),
            push_to_hub=False,   # we push manually below
        )

        FocalTrainer = make_focal_trainer(cw, gamma=GAMMA)
        trainer = FocalTrainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        _status = f"🏋️ Training ({EPOCHS} epochs, focal loss, A10G)…"
        _log("Starting training…")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        _log("Training complete.")

        # Final eval
        _status = "📊 Evaluating…"
        result = trainer.evaluate()
        _final_metrics = {k: round(v, 4) for k, v in result.items()}
        _log(f"Final metrics: {_final_metrics}")

        # Save label mapping alongside model
        output_dir.mkdir(exist_ok=True)
        (output_dir / "label_mapping.json").write_text(
            json.dumps({"label2id": label2id, "id2label": id2label}, indent=2)
        )

        # Push to Hub
        _status = "🚀 Pushing model to Hub…"
        _log(f"Pushing to {HUB_MODEL_ID}…")
        trainer.push_to_hub(
            repo_id=HUB_MODEL_ID,
            commit_message=f"trained roberta-base: macro_f1={_final_metrics.get('eval_macro_f1', '?')}",
            token=HF_TOKEN,
            blocking=True,
        )
        # Also upload label mapping
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(output_dir / "label_mapping.json"),
            path_in_repo="label_mapping.json",
            repo_id=HUB_MODEL_ID,
            commit_message="add label_mapping.json",
        )
        _log(f"✅ Model pushed to https://huggingface.co/{HUB_MODEL_ID}")
        _status = f"✅ Done! macro_f1={_final_metrics.get('eval_macro_f1', '?')}"
        _done = True

    except Exception as exc:
        import traceback
        msg = traceback.format_exc()
        _log(f"ERROR: {exc}\n{msg}")
        _status = f"❌ Error: {exc}"
        _done = True


# Start training in background thread immediately on Space startup
_thread = threading.Thread(target=train, daemon=True)
_thread.start()


# ── Gradio UI ────────────────────────────────────────────────────────────────
def get_status():
    log_text = "\n".join(_log_lines[-60:])        # last 60 lines
    chart = ""
    if _metrics_history:
        rows = ["| Step | Accuracy | Macro F1 |", "|---:|---:|---:|"]
        for i, m in enumerate(_metrics_history):
            rows.append(f"| {i+1} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
        chart = "\n".join(rows)
    final = ""
    if _final_metrics:
        final = "### Final metrics\n" + "\n".join(f"- **{k}**: {v}" for k, v in _final_metrics.items())
    return _status, log_text, chart, final


with gr.Blocks(title="Cipher Detective Trainer") as demo:
    gr.Markdown(
        "# 🧠 Cipher Detective — Training Job\n"
        "Training `roberta-base` on 81 cipher classes. "
        f"Target: [{HUB_MODEL_ID}](https://huggingface.co/{HUB_MODEL_ID})\n\n"
        "_Refresh the page or click **Refresh** to see latest progress._"
    )
    status_box = gr.Textbox(label="Status", value=_status, interactive=False)
    refresh_btn = gr.Button("🔄 Refresh")
    log_box     = gr.Textbox(label="Training log (last 60 lines)", lines=25, interactive=False)
    metrics_md  = gr.Markdown(label="Eval history")
    final_md    = gr.Markdown(label="Final metrics")

    def refresh():
        s, l, c, f = get_status()
        return s, l, c, f

    refresh_btn.click(refresh, outputs=[status_box, log_box, metrics_md, final_md])
    demo.load(refresh, outputs=[status_box, log_box, metrics_md, final_md], every=30)

demo.launch(server_name="0.0.0.0", server_port=7860)
