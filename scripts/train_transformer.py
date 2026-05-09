from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def make_weighted_trainer(class_weights_tensor):
    """Return a Trainer subclass that uses class-weighted cross-entropy loss."""

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = class_weights_tensor.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def make_focal_trainer(class_weights_tensor, gamma: float = 2.0):
    """Focal loss trainer: down-weights easy examples, focuses on hard ones.

    Combines class-weighting (for imbalance) with focal loss (for hard negatives).
    Recommended when the dataset has both class-imbalance AND many confusable pairs.
    """
    import torch.nn.functional as F

    class FocalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = class_weights_tensor.to(logits.device)
            # Standard weighted cross-entropy
            ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
            # Focal scaling: (1 - p_t)^gamma
            probs = F.softmax(logits, dim=-1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal = ((1 - pt) ** gamma) * ce
            loss = focal.mean()
            return (loss, outputs) if return_outputs else loss

    return FocalTrainer


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune a transformer for 81-class cipher identification."
    )
    ap.add_argument("--data", default="data/cipher_examples.jsonl")
    ap.add_argument(
        "--test-data", default=None,
        help="Separate JSONL eval file (e.g. blind split). "
             "If omitted, 15%% of --data is held out.",
    )
    ap.add_argument(
        "--model", default="roberta-base",
        help="Pre-trained model ID or local path. "
             "Smaller: distilroberta-base. Larger: roberta-large.",
    )
    ap.add_argument("--out", default="cipher_model")
    ap.add_argument("--epochs", type=float, default=10.0,
                    help="Training epochs. 10+ recommended for 81-class accuracy.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256,
                    help="Token length. 256 covers most cipher texts; raise for long ones.")
    ap.add_argument(
        "--weighted-loss", action="store_true", default=True,
        help="Use class-weighted cross-entropy (default: on). "
             "Essential given the 75:1 class-imbalance in the dataset.",
    )
    ap.add_argument(
        "--focal-loss", action="store_true",
        help="Use focal loss instead of plain weighted cross-entropy. "
             "Helps when many ciphers are statistically similar.",
    )
    ap.add_argument(
        "--lr", type=float, default=2e-5,
        help="Peak learning rate. 2e-5 works well for roberta-base; "
             "try 3e-5 for distilroberta.",
    )
    ap.add_argument("--warmup-ratio", type=float, default=0.06,
                    help="Fraction of total steps used for linear warmup.")
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="Label smoothing factor (0 = off). Helps with similar-class confusion.")
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps. Effective batch = batch-size × grad-accum.")
    ap.add_argument(
        "--early-stopping-patience", type=int, default=3,
        help="Stop training if macro_f1 doesn't improve for this many eval epochs (0 = off).",
    )
    ap.add_argument(
        "--push-to-hub", action="store_true",
        help="Push the trained model to the Hugging Face Hub after training.",
    )
    ap.add_argument(
        "--hub-model-id", default=None,
        help="Hub repo id for --push-to-hub (e.g. username/cipher-model). "
             "Required when --push-to-hub is set.",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.data)

    # Drop labels with fewer than 2 examples (can't stratify-split them).
    from collections import Counter
    label_counts = Counter(r["label"] for r in rows)
    dropped = {lbl for lbl, cnt in label_counts.items() if cnt < 2}
    if dropped:
        print(f"Dropping {len(dropped)} label(s) with <2 examples: {sorted(dropped)}")
        rows = [r for r in rows if r["label"] not in dropped]

    labels = sorted({r["label"] for r in rows})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Dataset: {len(rows):,} examples | {len(labels)} labels")
    print(f"Model: {args.model} | epochs: {args.epochs} | lr: {args.lr}")

    # Keep only the two columns needed for training.
    rows = [{"text": r["text"], "label_id": label2id[r["label"]]} for r in rows]

    if args.test_data:
        test_rows_raw = load_jsonl(args.test_data)
        test_rows = [
            {"text": r["text"], "label_id": label2id[r["label"]]}
            for r in test_rows_raw
            if r.get("label") in label2id
        ]
        train_rows = rows
        print(f"Using separate test file: {len(test_rows)} eval examples")
    else:
        train_rows, test_rows = train_test_split(
            rows,
            test_size=0.15,
            random_state=42,
            stratify=[r["label_id"] for r in rows],
        )

    ds_train = Dataset.from_list(train_rows)
    ds_test = Dataset.from_list(test_rows)

    tok = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    ds_train = ds_train.map(tokenize, batched=True)
    ds_test = ds_test.map(tokenize, batched=True)
    ds_train = ds_train.rename_column("label_id", "labels")
    ds_test = ds_test.rename_column("label_id", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Compute class weights for the weighted / focal loss trainer.
    train_label_ids = [r["label_id"] for r in train_rows]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(labels)),
        y=train_label_ids,
    )
    # Cap extreme weights to prevent instability on very rare classes.
    class_weights = np.clip(class_weights, 0.1, 20.0)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Class weights — min: {weights_tensor.min():.2f} max: {weights_tensor.max():.2f}")

    training_args = TrainingArguments(
        output_dir=args.out,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        # Mixed-precision: speeds up training on modern GPUs
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        # Hub push (only active when --push-to-hub is passed)
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    if args.focal_loss:
        print("Using focal loss (with class weighting)")
        TrainerClass = make_focal_trainer(weights_tensor)
    elif args.weighted_loss:
        print("Using class-weighted cross-entropy loss")
        TrainerClass = make_weighted_trainer(weights_tensor)
    else:
        print("Using standard cross-entropy loss (no class weighting)")
        TrainerClass = Trainer

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        print(f"Early stopping: patience={args.early_stopping_patience} epochs")

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=callbacks or None,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    if args.push_to_hub:
        print(f"Pushing model to Hub: {args.hub_model_id}")
        trainer.push_to_hub()

    out_path = Path(args.out)
    (out_path / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_path / "label_mapping.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {args.out}")
    print(f"Accuracy: {metrics.get('eval_accuracy', 0):.3f}")
    print(f"Macro F1: {metrics.get('eval_macro_f1', 0):.3f}")


if __name__ == "__main__":
    main()
