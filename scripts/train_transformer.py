from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cipher_examples.jsonl")
    ap.add_argument("--test-data", default=None,
                    help="Separate JSONL file to use as eval set (e.g. blind split). "
                         "If omitted, 20%% of --data is held out.")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--out", default="cipher_model")
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--weighted-loss", action="store_true",
                    help="Use class-weighted cross-entropy to handle imbalance.")
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

    # Keep only the two columns needed for training.  The rich dataset schema
    # contains heterogeneous dict fields (e.g. `key`) that would cause PyArrow
    # schema-inference failures inside Dataset.from_list().
    rows = [{"text": r["text"], "label_id": label2id[r["label"]]} for r in rows]

    if args.test_data:
        test_rows_raw = load_jsonl(args.test_data)
        # Apply same label vocabulary (ignore test labels not in train)
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
            test_size=0.2,
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

    training_args = TrainingArguments(
        output_dir=args.out,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
    )

    if args.weighted_loss:
        train_label_ids = [r["label_id"] for r in train_rows]
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(labels)),
            y=train_label_ids,
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        print(f"Class weighting enabled — weight range: [{weights_tensor.min():.2f}, {weights_tensor.max():.2f}]")
        TrainerClass = make_weighted_trainer(weights_tensor)
    else:
        TrainerClass = Trainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    out_path = Path(args.out)
    (out_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_path / "label_mapping.json").write_text(json.dumps({"label2id": label2id, "id2label": id2label}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()
