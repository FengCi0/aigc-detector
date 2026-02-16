#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练 Transformer AIGC 检测分支（用于与轻量主模型融合）。
"""

import argparse
import inspect
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except Exception as exc:  # pragma: no cover - 运行时依赖检查
    raise RuntimeError(
        "训练 Transformer 需要安装 torch/datasets/transformers/accelerate。"
        f" 当前导入失败: {exc}"
    ) from exc

try:
    from backend.utils.text_processing import preprocess_text, setup_logging
except ImportError:  # 兼容在 backend 目录运行
    from utils.text_processing import preprocess_text, setup_logging


logger = setup_logging()


def load_text_files(directory: Path, max_files: Optional[int], min_length: int) -> List[str]:
    files = list(directory.rglob("*.txt"))
    if not files:
        return []
    if max_files is not None and max_files > 0 and len(files) > max_files:
        files = random.sample(files, max_files)

    texts: List[str] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = preprocess_text(content, preserve_newlines=False)
            if len(cleaned) >= min_length:
                texts.append(cleaned)
        except Exception as exc:
            logger.warning("读取失败: %s (%s)", path, exc)
    return texts


def build_splits(
    ai_texts: List[str],
    human_texts: List[str],
    test_ratio: float,
    eval_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[object]], Dict[str, List[object]], Dict[str, List[object]]]:
    texts = np.array(ai_texts + human_texts, dtype=object)
    labels = np.array([1] * len(ai_texts) + [0] * len(human_texts), dtype=int)
    indices = np.arange(len(labels))

    train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=eval_ratio,
        random_state=seed,
        stratify=labels[train_val_idx],
    )

    def subset(idxs: np.ndarray) -> Dict[str, List[object]]:
        return {
            "text": texts[idxs].tolist(),
            "label": labels[idxs].tolist(),
        }

    return subset(train_idx), subset(val_idx), subset(test_idx)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def evaluate_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = _softmax(np.asarray(logits, dtype=float))
    y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    y_pred = np.argmax(probs, axis=1)
    y_true = np.asarray(labels, dtype=int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def archive_previous_model(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    if not (output_dir / "config.json").exists():
        return None
    archive_root = output_dir.parent / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    archived = archive_root / f"{output_dir.name}.{stamp}"
    if archived.exists():
        shutil.rmtree(archived)
    shutil.move(str(output_dir), str(archived))
    return str(archived)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Transformer AIGC 检测分支")
    parser.add_argument("--ai-data", required=True)
    parser.add_argument("--human-data", required=True)
    parser.add_argument("--output-dir", default="data/models/transformer_detector")
    parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--archive-previous", action="store_true", help="训练前备份旧 Transformer 模型到 data/models/archive/")
    args = parser.parse_args()

    os_max_files: Optional[int] = None
    if args.max_files and args.max_files > 0:
        os_max_files = int(args.max_files)

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    ai_dir = Path(args.ai_data)
    human_dir = Path(args.human_data)
    if not ai_dir.exists() or not human_dir.exists():
        raise RuntimeError(f"训练目录不存在: ai={ai_dir}, human={human_dir}")

    ai_texts = load_text_files(ai_dir, max_files=os_max_files, min_length=args.min_length)
    human_texts = load_text_files(human_dir, max_files=os_max_files, min_length=args.min_length)
    if len(ai_texts) < 50 or len(human_texts) < 50:
        raise RuntimeError(
            f"Transformer 训练数据不足（每类至少 50 条）。当前 ai={len(ai_texts)}, human={len(human_texts)}"
        )

    train_data, val_data, test_data = build_splits(
        ai_texts=ai_texts,
        human_texts=human_texts,
        test_ratio=args.test_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    logger.info(
        "Transformer 数据集划分: train=%d val=%d test=%d",
        len(train_data["label"]),
        len(val_data["label"]),
        len(test_data["label"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    )

    train_ds = Dataset.from_dict(train_data)
    val_ds = Dataset.from_dict(val_data)
    test_ds = Dataset.from_dict(test_data)

    def tokenize_fn(batch: Dict[str, List[object]]) -> Dict[str, object]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max(64, int(args.max_length)),
            padding=False,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])
    test_ds = test_ds.remove_columns(["text"])

    run_dir = Path(args.output_dir).parent / "_transformer_runs" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    use_fp16 = bool(torch.cuda.is_available() and not args.no_fp16)
    logger.info("Transformer 训练设备: cuda=%s fp16=%s", torch.cuda.is_available(), use_fp16)

    training_kwargs = {
        "output_dir": str(run_dir),
        "num_train_epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate),
        "per_device_train_batch_size": int(args.batch_size),
        "per_device_eval_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": max(int(args.gradient_accumulation_steps), 1),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": 20,
        "report_to": [],
        "fp16": use_fp16,
        "seed": int(args.seed),
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        return evaluate_logits(np.asarray(logits), np.asarray(labels))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_prediction = trainer.predict(test_ds)
    test_metrics = evaluate_logits(
        logits=np.asarray(test_prediction.predictions),
        labels=np.asarray(test_prediction.label_ids),
    )

    output_dir = Path(args.output_dir)
    previous_backup = archive_previous_model(output_dir) if args.archive_previous else None
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_mode": "transformer_finetune",
        "base_model": args.base_model,
        "output_dir": str(output_dir.resolve()),
        "previous_backup": previous_backup,
        "dataset": {
            "ai_data_dir": str(ai_dir.resolve()),
            "human_data_dir": str(human_dir.resolve()),
            "ai_count": len(ai_texts),
            "human_count": len(human_texts),
            "train_size": len(train_data["label"]),
            "val_size": len(val_data["label"]),
            "test_size": len(test_data["label"]),
            "min_length": int(args.min_length),
        },
        "hyperparameters": {
            "epochs": float(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "max_length": int(args.max_length),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "fp16": use_fp16,
            "seed": int(args.seed),
        },
        "validation_metrics": {
            "accuracy": float(eval_metrics.get("eval_accuracy", 0.0)),
            "precision": float(eval_metrics.get("eval_precision", 0.0)),
            "recall": float(eval_metrics.get("eval_recall", 0.0)),
            "f1": float(eval_metrics.get("eval_f1", 0.0)),
            "roc_auc": float(eval_metrics.get("eval_roc_auc", 0.0)),
            "loss": float(eval_metrics.get("eval_loss", 0.0)),
        },
        "test_metrics": test_metrics,
        "recommended_env": {
            "AIGC_ENABLE_TRANSFORMER": "1",
            "AIGC_TRANSFORMER_MODEL_ID": str(output_dir.resolve()),
            "AIGC_TRANSFORMER_APPLY_TO": "all",
            "AIGC_TRANSFORMER_WEIGHT": "1.0",
        },
    }

    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Transformer 模型已保存: %s", output_dir)
    logger.info("Transformer 元数据已保存: %s", metadata_path)
    logger.info("Transformer 验证集指标: %s", metadata["validation_metrics"])
    logger.info("Transformer 测试集指标: %s", metadata["test_metrics"])


if __name__ == "__main__":
    main()
