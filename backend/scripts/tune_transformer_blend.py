#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""为现有轻量模型搜索 Transformer 融合权重与阈值。"""

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.features import FeatureExtractor
from backend.models.transformer_branch import TransformerAIGCScorer
from backend.utils.text_processing import preprocess_text


def _safe_prob(model, x) -> Optional[float]:
    if model is None:
        return None
    try:
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba([x])[0][1])
        if hasattr(model, "decision_function"):
            decision = float(model.decision_function([x])[0])
            return 1.0 / (1.0 + np.exp(-decision))
        return None
    except Exception:
        return None


def _safe_text_prob(model, text: str) -> Optional[float]:
    if model is None:
        return None
    try:
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba([text])[0][1])
        if hasattr(model, "decision_function"):
            decision = float(model.decision_function([text])[0])
            return 1.0 / (1.0 + np.exp(-decision))
        return None
    except Exception:
        return None


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return out


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_th = 0.5
    best = _evaluate(y_true, y_prob, best_th)
    for th in np.linspace(0.2, 0.8, 121):
        metrics = _evaluate(y_true, y_prob, float(th))
        if metrics["f1"] > best["f1"] or (abs(metrics["f1"] - best["f1"]) < 1e-12 and metrics["accuracy"] > best["accuracy"]):
            best_th = float(th)
            best = metrics
    return best_th, best


def _load_texts(ai_dir: str, human_dir: str, max_samples: int, seed: int, min_text_length: int) -> Tuple[List[str], np.ndarray]:
    ai_files = glob.glob(f"{ai_dir}/*.txt")
    human_files = glob.glob(f"{human_dir}/*.txt")
    random.seed(seed)
    if max_samples > 0:
        ai_files = random.sample(ai_files, min(max_samples, len(ai_files)))
        human_files = random.sample(human_files, min(max_samples, len(human_files)))

    texts: List[str] = []
    labels: List[int] = []
    for p in ai_files:
        text = preprocess_text(Path(p).read_text(encoding="utf-8", errors="ignore"), preserve_newlines=True)
        if len(text) >= min_text_length:
            texts.append(text)
            labels.append(1)
    for p in human_files:
        text = preprocess_text(Path(p).read_text(encoding="utf-8", errors="ignore"), preserve_newlines=True)
        if len(text) >= min_text_length:
            texts.append(text)
            labels.append(0)

    merged = list(zip(texts, labels))
    random.shuffle(merged)
    if not merged:
        return [], np.array([], dtype=int)
    texts, labels = zip(*merged)
    return list(texts), np.array(labels, dtype=int)


def _parse_weight_grid(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            v = float(item)
            values.append(float(min(max(v, 0.0), 1.0)))
        except ValueError:
            continue
    if not values:
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="搜索 Transformer 融合权重")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ai-dir", required=True)
    parser.add_argument("--human-dir", required=True)
    parser.add_argument("--transformer-model-id", required=True)
    parser.add_argument("--transformer-device", default="auto")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--min-text-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-grid", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    feature_model = bundle.get("feature_classifier") or bundle.get("classifier")
    text_model = bundle.get("text_classifier")
    metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    text_model_weight = float(metadata.get("text_model_weight", 0.75))
    base_threshold = float(metadata.get("recommended_threshold", 0.47))

    scorer = TransformerAIGCScorer(
        model_id=args.transformer_model_id,
        device=args.transformer_device,
        max_length=512,
        chunk_size_chars=600,
    )
    if not scorer.available:
        raise RuntimeError(f"Transformer 模型不可用: {scorer.reason}")

    texts, labels = _load_texts(
        ai_dir=args.ai_dir,
        human_dir=args.human_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        min_text_length=args.min_text_length,
    )
    if len(texts) < 100:
        raise RuntimeError(f"样本不足，当前有效样本数={len(texts)}")

    extractor = FeatureExtractor()
    feature_vectors = [extractor.extract_features(t) for t in texts]

    classic_probs: List[float] = []
    trans_probs: List[float] = []
    for text, x in zip(texts, feature_vectors):
        fp = _safe_prob(feature_model, x)
        tp = _safe_text_prob(text_model, text)
        if fp is None and tp is None:
            classic = 0.5
        elif fp is None:
            classic = float(tp)
        elif tp is None:
            classic = float(fp)
        else:
            classic = (1.0 - text_model_weight) * float(fp) + text_model_weight * float(tp)
        classic_probs.append(float(min(max(classic, 0.0), 1.0)))

        trans = scorer.score_text(text)
        trans_prob = 0.5 if trans is None else float(trans.probability)
        trans_probs.append(float(min(max(trans_prob, 0.0), 1.0)))

    y_true = np.array(labels, dtype=int)
    classic_prob_arr = np.array(classic_probs, dtype=float)
    trans_prob_arr = np.array(trans_probs, dtype=float)

    weight_grid = _parse_weight_grid(args.weight_grid)
    best = None
    rows = []
    for w in weight_grid:
        probs = (1.0 - w) * classic_prob_arr + w * trans_prob_arr
        threshold, metrics = _find_best_threshold(y_true, probs)
        row = {
            "transformer_weight": float(w),
            "best_threshold": float(threshold),
            **metrics,
        }
        rows.append(row)
        if best is None or row["f1"] > best["f1"] or (abs(row["f1"] - best["f1"]) < 1e-12 and row["accuracy"] > best["accuracy"]):
            best = row

    assert best is not None
    baseline_metrics = _evaluate(y_true, classic_prob_arr, threshold=base_threshold)
    report = {
        "model_path": args.model_path,
        "transformer_model_id": args.transformer_model_id,
        "sample_count": int(len(y_true)),
        "base_threshold": float(base_threshold),
        "baseline": baseline_metrics,
        "grid": rows,
        "best": best,
        "recommended_env": {
            "AIGC_ENABLE_TRANSFORMER": "1",
            "AIGC_TRANSFORMER_MODEL_ID": args.transformer_model_id,
            "AIGC_TRANSFORMER_WEIGHT": f"{best['transformer_weight']:.3f}",
            "AIGC_SCORE_THRESHOLD": f"{best['best_threshold']:.3f}",
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
