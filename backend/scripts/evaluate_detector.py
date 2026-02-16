#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""评估检测器在本地 AI/Human 文本目录上的效果。"""

import argparse
import glob
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.detector import AIGCDetector


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    labels = np.asarray(y_true, dtype=int)
    if probs.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = float(probs.size)
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        if i == bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        count = int(np.sum(mask))
        if count <= 0:
            continue
        avg_conf = float(np.mean(probs[mask]))
        avg_acc = float(np.mean(labels[mask]))
        ece += abs(avg_conf - avg_acc) * (count / n)
    return float(ece)


def load_texts(ai_dir: str, human_dir: str, max_samples: int, seed: int) -> Tuple[List[str], List[int]]:
    ai_files = glob.glob(f"{ai_dir}/*.txt")
    human_files = glob.glob(f"{human_dir}/*.txt")

    random.seed(seed)
    if max_samples:
        ai_files = random.sample(ai_files, min(max_samples, len(ai_files)))
        human_files = random.sample(human_files, min(max_samples, len(human_files)))

    texts: List[str] = []
    labels: List[int] = []
    for p in ai_files:
        texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
        labels.append(1)
    for p in human_files:
        texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
        labels.append(0)

    merged = list(zip(texts, labels))
    if not merged:
        return [], []
    random.shuffle(merged)
    texts, labels = zip(*merged)
    return list(texts), list(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 AIGC 检测器")
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--ai-dir", default="data/dataset/ai", help="AI 文本目录")
    parser.add_argument("--human-dir", default="data/dataset/human", help="Human 文本目录")
    parser.add_argument("--max-samples", type=int, default=2000, help="每类最多抽样数量")
    parser.add_argument("--min-text-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="", help="可选，评估报告输出 JSON")
    args = parser.parse_args()

    texts, labels = load_texts(args.ai_dir, args.human_dir, args.max_samples, args.seed)
    detector = AIGCDetector(model_path=args.model_path, min_text_length=args.min_text_length)

    scores = []
    preds = []
    used_labels = []
    skipped = 0
    for text, label in zip(texts, labels):
        try:
            result = detector.analyze(text, include_details=False)
            score = float(result["aigc_score"]) / 100.0
            pred = 1 if score >= (float(result["score_threshold"]) / 100.0) else 0
            scores.append(score)
            preds.append(pred)
            used_labels.append(label)
        except ValueError:
            skipped += 1

    y_true = np.array(used_labels, dtype=int)
    y_pred = np.array(preds, dtype=int)
    y_score = np.array(scores, dtype=float)

    if y_true.size == 0:
        report = {
            "model_path": args.model_path,
            "model_mode": detector.model_mode,
            "model_loaded": detector.model_loaded,
            "sample_count_used": 0,
            "sample_count_skipped": int(skipped),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
            "ece": None,
            "brier": None,
            "log_loss": None,
            "error": "no_valid_samples",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    has_two_classes = len(np.unique(y_true)) > 1

    report = {
        "model_path": args.model_path,
        "model_mode": detector.model_mode,
        "model_loaded": detector.model_loaded,
        "sample_count_used": int(len(y_true)),
        "sample_count_skipped": int(skipped),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_two_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_two_classes else None,
        "ece": expected_calibration_error(y_true, y_score),
        "brier": float(np.mean((y_score - y_true) ** 2)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
