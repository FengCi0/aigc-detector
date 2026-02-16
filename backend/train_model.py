#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练AIGC检测模型脚本（泛化增强版）
"""

import argparse
import json
import math
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from backend.models.features import FeatureExtractor
    from backend.utils.text_processing import preprocess_text, setup_logging
except ImportError:  # 兼容在backend目录执行脚本
    from models.features import FeatureExtractor
    from utils.text_processing import preprocess_text, setup_logging


logger = setup_logging()
EPS = 1e-6
DEFAULT_MODEL_OUTPUT = Path("data/models/aigc_detector_model.joblib")


def load_text_files(directory: str, pattern: str = "*.txt", max_files: Optional[int] = None, min_length: int = 50) -> List[str]:
    folder = Path(directory)
    if not folder.exists():
        logger.error("目录不存在: %s", directory)
        return []

    files = list(folder.rglob(pattern))
    if not files:
        logger.warning("目录中没有匹配文件: %s (%s)", directory, pattern)
        return []

    if max_files and len(files) > max_files:
        files = random.sample(files, max_files)

    texts: List[str] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = preprocess_text(content, preserve_newlines=True)
            if len(cleaned) >= min_length:
                texts.append(cleaned)
        except Exception as exc:
            logger.warning("读取失败: %s (%s)", path, exc)

    logger.info("加载文本: %s -> %d", directory, len(texts))
    return texts


def build_feature_candidates(seed: int) -> Dict[str, object]:
    return {
        "feature_logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
            ]
        ),
        "feature_rf": RandomForestClassifier(
            n_estimators=360,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
        ),
        "feature_gb": GradientBoostingClassifier(random_state=seed),
    }


def build_text_model(seed: int, max_features: int = 80000) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
        ]
    )


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    return metrics


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


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=float)
    probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if labels.size == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=float)
    probs = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    if labels.size == 0:
        return 0.0
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def _to_logits(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "ece": expected_calibration_error(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "log_loss": log_loss_score(y_true, y_prob),
        "extreme_rate": float(np.mean((y_prob <= 0.01) | (y_prob >= 0.99))),
    }


def fit_isotonic_calibrator(y_true: np.ndarray, y_prob: np.ndarray, min_samples: int = 200) -> Optional[Dict[str, object]]:
    probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    labels = np.asarray(y_true, dtype=int)
    if probs.size < min_samples or len(np.unique(labels)) < 2:
        return None
    try:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(probs, labels)
        return {"method": "isotonic_regression", "model": model}
    except Exception as exc:
        logger.warning("Isotonic 校准失败，跳过: %s", exc)
        return None


def fit_temperature_calibrator(
    y_true: np.ndarray, y_prob: np.ndarray, min_samples: int = 200
) -> Optional[Dict[str, object]]:
    probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    labels = np.asarray(y_true, dtype=int)
    if probs.size < min_samples or len(np.unique(labels)) < 2:
        return None

    logits = _to_logits(probs)

    def objective(log_t: np.ndarray) -> float:
        temperature = float(np.exp(log_t[0]))
        calibrated = _sigmoid(logits / temperature)
        return log_loss_score(labels, calibrated)

    try:
        result = minimize(
            objective,
            x0=np.array([0.0], dtype=float),
            method="L-BFGS-B",
            bounds=[(math.log(0.05), math.log(20.0))],
        )
        if not result.success:
            return None
        return {"method": "temperature_scaling", "temperature": float(np.exp(result.x[0]))}
    except Exception as exc:
        logger.warning("Temperature 校准失败，跳过: %s", exc)
        return None


def select_probability_calibrator(
    y_true: np.ndarray, y_prob: np.ndarray, min_samples: int = 200, seed: int = 42
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    labels = np.asarray(y_true, dtype=int)
    full_baseline = calibration_metrics(labels, probs)

    report: Dict[str, object] = {
        "enabled": False,
        "method": "none",
        "val_ece_before": full_baseline["ece"],
        "val_ece_after": full_baseline["ece"],
        "val_brier_before": full_baseline["brier"],
        "val_brier_after": full_baseline["brier"],
        "val_log_loss_before": full_baseline["log_loss"],
        "val_log_loss_after": full_baseline["log_loss"],
        "candidates": {
            "none": {
                "enabled": False,
                "eval": full_baseline,
                "full": full_baseline,
            }
        },
    }

    if probs.size < min_samples or len(np.unique(labels)) < 2:
        return None, report

    indices = np.arange(labels.size)
    fit_idx, eval_idx = train_test_split(indices, test_size=0.5, random_state=seed, stratify=labels)
    y_fit = labels[fit_idx]
    p_fit = probs[fit_idx]
    y_eval = labels[eval_idx]
    p_eval = probs[eval_idx]
    eval_baseline = calibration_metrics(y_eval, p_eval)
    report["candidates"]["none"]["eval"] = eval_baseline

    best_method = "none"
    best_calibrator: Optional[Dict[str, object]] = None
    best_eval = eval_baseline

    def try_candidate(name: str, calibrator: Optional[Dict[str, object]]) -> None:
        nonlocal best_method, best_calibrator, best_eval
        if calibrator is None:
            report["candidates"][name] = {"enabled": False}
            return
        eval_probs = apply_probability_calibrator(calibrator, p_eval)
        eval_metrics = calibration_metrics(y_eval, eval_probs)
        full_probs = apply_probability_calibrator(calibrator, probs)
        full_metrics = calibration_metrics(labels, full_probs)
        enabled = True
        report["candidates"][name] = {
            "enabled": enabled,
            "eval": eval_metrics,
            "full": full_metrics,
        }
        # 先比Brier，再比ECE，最后比log_loss；并且避免极端压缩明显恶化
        candidate_key = (eval_metrics["brier"], eval_metrics["ece"], eval_metrics["log_loss"])
        best_key = (best_eval["brier"], best_eval["ece"], best_eval["log_loss"])
        if candidate_key < best_key and eval_metrics["extreme_rate"] <= max(best_eval["extreme_rate"] * 1.3, 0.2):
            best_method = name
            best_calibrator = calibrator
            best_eval = eval_metrics

    iso = fit_isotonic_calibrator(y_fit, p_fit, min_samples=max(min_samples // 2, 80))
    try_candidate("isotonic_regression", iso)

    temp = fit_temperature_calibrator(y_fit, p_fit, min_samples=max(min_samples // 2, 80))
    try_candidate("temperature_scaling", temp)

    if best_method == "none":
        return None, report

    # 最终模型：temperature 可在全量验证集重拟合；isotonic 保持子集拟合避免过拟合
    if best_method == "temperature_scaling":
        refit = fit_temperature_calibrator(labels, probs, min_samples=min_samples)
        if refit is not None:
            best_calibrator = refit
    elif best_method == "isotonic_regression":
        if best_calibrator is None:
            best_calibrator = fit_isotonic_calibrator(y_fit, p_fit, min_samples=max(min_samples // 2, 80))

    final_probs = apply_probability_calibrator(best_calibrator, probs) if best_calibrator is not None else probs
    final_metrics = calibration_metrics(labels, final_probs)
    report["enabled"] = True
    report["method"] = best_method
    report["val_ece_after"] = final_metrics["ece"]
    report["val_brier_after"] = final_metrics["brier"]
    report["val_log_loss_after"] = final_metrics["log_loss"]
    if best_method == "temperature_scaling" and isinstance(best_calibrator, dict):
        report["temperature"] = float(best_calibrator.get("temperature", 1.0))
    return best_calibrator, report


def apply_probability_calibrator(calibrator: Optional[Dict[str, object]], y_prob: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if calibrator is None:
        return probs

    method = str(calibrator.get("method", "none"))
    if method == "isotonic_regression":
        model = calibrator.get("model")
        if model is None:
            return probs
        calibrated = np.asarray(model.predict(probs), dtype=float)
        return np.clip(calibrated, 0.0, 1.0)
    if method == "temperature_scaling":
        temperature = float(calibrator.get("temperature", 1.0))
        temperature = max(temperature, 0.05)
        calibrated = _sigmoid(_to_logits(probs) / temperature)
        return np.clip(calibrated, 0.0, 1.0)
    return probs


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = evaluate_binary(y_true, y_prob, threshold=0.5)
    best_f1 = best_metrics["f1"]
    best_acc = best_metrics["accuracy"]
    for threshold in np.linspace(0.3, 0.7, 81):
        metrics = evaluate_binary(y_true, y_prob, threshold=float(threshold))
        f1 = metrics["f1"]
        acc = metrics["accuracy"]
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and acc > best_acc):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = f1
            best_acc = acc
    return best_threshold, best_metrics


def train_best_feature_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int
) -> Tuple[str, object, np.ndarray, Dict[str, Dict[str, float]]]:
    candidates = build_feature_candidates(seed)
    best_name = ""
    best_model = None
    best_f1 = -1.0
    best_val_prob = None
    metrics_map: Dict[str, Dict[str, float]] = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        if not hasattr(model, "predict_proba"):
            continue
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, metrics = find_best_threshold(y_val, val_prob)
        metrics["best_threshold"] = threshold
        metrics_map[name] = metrics
        logger.info("候选特征模型 %s 验证集: %s", name, metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = model
            best_val_prob = val_prob

    if best_model is None or best_val_prob is None:
        raise RuntimeError("特征模型训练失败")

    return best_name, best_model, best_val_prob, metrics_map


def find_best_blend(
    y_val: np.ndarray, feature_val_prob: np.ndarray, text_val_prob: np.ndarray
) -> Tuple[float, float, Dict[str, float], Dict[str, Dict[str, float]]]:
    best_weight_text = 0.0
    best_threshold = 0.5
    best_metrics = evaluate_binary(y_val, feature_val_prob)
    best_f1 = best_metrics["f1"]
    best_acc = best_metrics["accuracy"]
    grid_report: Dict[str, Dict[str, float]] = {}

    for w_text in np.linspace(0.0, 1.0, 21):
        blended = (1.0 - w_text) * feature_val_prob + w_text * text_val_prob
        threshold, metrics = find_best_threshold(y_val, blended)
        key = f"text_weight_{w_text:.2f}"
        metrics_with_cfg = dict(metrics)
        metrics_with_cfg["threshold"] = threshold
        grid_report[key] = metrics_with_cfg
        f1 = metrics["f1"]
        acc = metrics["accuracy"]
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-12 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_weight_text = float(w_text)
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_weight_text, best_threshold, best_metrics, grid_report


def compute_short_text_threshold(
    y_val: np.ndarray,
    blend_val_prob: np.ndarray,
    val_texts: List[str],
    short_text_length_upper: int,
    target_recall: float,
) -> Tuple[float, Dict[str, float], int]:
    lengths = np.array([len(t) for t in val_texts], dtype=int)
    mask = lengths <= short_text_length_upper
    count = int(np.sum(mask))
    if count < 60:
        threshold, metrics = find_best_threshold(y_val, blend_val_prob)
        return threshold, metrics, count
    y_short = y_val[mask]
    if len(np.unique(y_short)) < 2:
        threshold, metrics = find_best_threshold(y_val, blend_val_prob)
        return threshold, metrics, count
    y_prob_short = blend_val_prob[mask]
    best = None
    fallback = None
    for threshold in np.linspace(0.3, 0.7, 81):
        metrics = evaluate_binary(y_short, y_prob_short, threshold=float(threshold))
        recall = metrics["recall"]
        f1 = metrics["f1"]
        acc = metrics["accuracy"]
        candidate = (float(threshold), metrics, f1, acc, recall)
        if fallback is None or f1 > fallback[2] or (abs(f1 - fallback[2]) < 1e-12 and acc > fallback[3]):
            fallback = candidate
        if recall >= target_recall:
            if best is None or f1 > best[2] or (abs(f1 - best[2]) < 1e-12 and acc > best[3]):
                best = candidate

    chosen = best if best is not None else fallback
    assert chosen is not None
    return chosen[0], chosen[1], count


def _split_sentences_for_augmentation(text: str) -> List[str]:
    merged = text.replace("\n", " ").strip()
    if not merged:
        return []
    parts = re.split(r"(?<=[。！？!?；;])", merged)
    sentences = [p.strip() for p in parts if p and p.strip()]
    if len(sentences) <= 1:
        # 退化场景：没有明显句末标点时，按逗号/顿号尝试切分
        clauses = [p.strip() for p in re.split(r"[，,、]", merged) if p and p.strip()]
        return clauses
    return sentences


def _augment_by_random_crop(
    text: str,
    rnd: random.Random,
    min_len: int,
    max_len: int,
) -> Optional[str]:
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) < min_len:
        return None
    upper = min(max_len, len(cleaned))
    lower = min(min_len, upper)
    if lower <= 0 or upper <= 0:
        return None
    length = rnd.randint(lower, upper)
    start_max = max(0, len(cleaned) - length)
    start = rnd.randint(0, start_max) if start_max > 0 else 0
    short = cleaned[start : start + length].strip()
    if len(short) < min_len:
        return None
    return short


def _augment_by_sentence_deletion(
    text: str,
    rnd: random.Random,
    drop_prob: float,
    min_len: int,
) -> Optional[str]:
    sentences = _split_sentences_for_augmentation(text)
    if len(sentences) <= 1:
        return None

    kept: List[str] = []
    for sentence in sentences:
        if rnd.random() > drop_prob:
            kept.append(sentence)
    if not kept:
        kept = [rnd.choice(sentences)]

    augmented = "".join(kept).strip()
    if len(augmented) < min_len:
        return None
    if augmented == text.replace("\n", " ").strip():
        return None
    return augmented


def augment_short_texts(
    texts: List[str],
    labels: np.ndarray,
    ratio: float,
    min_len: int = 70,
    max_len: int = 180,
    sentence_drop_prob: float = 0.25,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, int]:
    if ratio <= 0:
        return texts, labels, 0

    rnd = random.Random(seed)
    augmented_texts = list(texts)
    augmented_labels = list(labels.tolist())
    n = len(texts)
    if n == 0:
        return texts, labels, 0
    target = max(1, int(n * ratio))
    indices = list(range(n))
    rnd.shuffle(indices)
    generated = 0
    attempts = 0
    max_attempts = max(n * 6, target * 8)

    while generated < target and attempts < max_attempts:
        idx = indices[attempts % n]
        text = texts[idx]

        if attempts % 2 == 0:
            candidate = _augment_by_random_crop(text, rnd, min_len=min_len, max_len=max_len)
            if candidate is None:
                candidate = _augment_by_sentence_deletion(
                    text, rnd, drop_prob=sentence_drop_prob, min_len=min_len
                )
        else:
            candidate = _augment_by_sentence_deletion(
                text, rnd, drop_prob=sentence_drop_prob, min_len=min_len
            )
            if candidate is None:
                candidate = _augment_by_random_crop(text, rnd, min_len=min_len, max_len=max_len)

        if candidate:
            augmented_texts.append(candidate)
            augmented_labels.append(int(labels[idx]))
            generated += 1
        attempts += 1

    return augmented_texts, np.array(augmented_labels, dtype=int), generated


def _load_previous_iteration(output_path: Path) -> int:
    metadata_path = output_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return 0
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return int(payload.get("training_iteration", 0))
    except Exception:
        return 0


def _archive_existing_model(output_path: Path) -> Optional[str]:
    if not output_path.exists():
        return None
    archive_dir = output_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    archived_model = archive_dir / f"{output_path.stem}.{stamp}.joblib"
    shutil.copy2(output_path, archived_model)
    metadata_path = output_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        archived_meta = archive_dir / f"{output_path.stem}.{stamp}.metadata.json"
        shutil.copy2(metadata_path, archived_meta)
    return str(archived_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练AIGC检测模型（双模型融合）")
    parser.add_argument("--ai-data", required=True, help="AI生成文本目录")
    parser.add_argument("--human-data", required=True, help="人工文本目录")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_OUTPUT), help="模型输出路径（默认固定主模型路径）")
    parser.add_argument("--file-pattern", default="*.txt", help="文件匹配模式（递归搜索）")
    parser.add_argument("--max-files", type=int, default=None, help="每类最大文件数")
    parser.add_argument("--min-length", type=int, default=50, help="最短文本长度")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--val-size", type=float, default=0.2, help="验证集比例（在训练池中切分）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--text-max-features", type=int, default=80000, help="文本模型TF-IDF特征上限")
    parser.add_argument("--short-text-length-upper", type=int, default=160, help="短文本阈值上界")
    parser.add_argument("--short-text-target-recall", type=float, default=0.93, help="短文本AI目标召回率")
    parser.add_argument("--augment-short-ratio", type=float, default=0.3, help="训练时短文本增强比例(0~1)")
    parser.add_argument(
        "--augment-sentence-drop-prob",
        type=float,
        default=0.25,
        help="多尺度增强中句子随机删除概率(0~1)",
    )
    parser.add_argument("--calibration-min-samples", type=int, default=200, help="启用校准的最小验证样本数")
    parser.add_argument("--archive-previous", action="store_true", help="训练前备份旧模型到 data/models/archive/")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ai_texts = load_text_files(args.ai_data, args.file_pattern, args.max_files, args.min_length)
    human_texts = load_text_files(args.human_data, args.file_pattern, args.max_files, args.min_length)
    if len(ai_texts) < 20 or len(human_texts) < 20:
        raise RuntimeError("每类至少需要20条有效样本，当前数据不足")

    texts = np.array(ai_texts + human_texts, dtype=object)
    labels = np.array([1] * len(ai_texts) + [0] * len(human_texts), dtype=int)

    extractor = FeatureExtractor()
    X_features = np.array([extractor.extract_features(t) for t in texts], dtype=float)

    all_indices = np.arange(len(labels))
    train_pool_idx, test_idx = train_test_split(
        all_indices, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_pool_idx, test_size=args.val_size, random_state=args.seed, stratify=labels[train_pool_idx]
    )

    X_train = X_features[train_idx]
    y_train = labels[train_idx]
    X_val = X_features[val_idx]
    y_val = labels[val_idx]
    X_test = X_features[test_idx]
    y_test = labels[test_idx]

    train_texts = texts[train_idx].tolist()
    val_texts = texts[val_idx].tolist()
    test_texts = texts[test_idx].tolist()

    feature_name, feature_model, feature_val_prob, feature_val_metrics = train_best_feature_model(
        X_train, y_train, X_val, y_val, args.seed
    )

    train_texts_aug, y_train_aug, aug_count = augment_short_texts(
        texts=train_texts,
        labels=y_train,
        ratio=max(0.0, min(args.augment_short_ratio, 1.0)),
        sentence_drop_prob=max(0.0, min(args.augment_sentence_drop_prob, 1.0)),
        seed=args.seed,
    )
    logger.info(
        "短文本/多尺度增强完成: 原始=%d 增强后=%d 新增=%d",
        len(train_texts),
        len(train_texts_aug),
        aug_count,
    )
    text_model = build_text_model(args.seed, args.text_max_features)
    text_model.fit(train_texts_aug, y_train_aug)
    text_val_prob = text_model.predict_proba(val_texts)[:, 1]
    text_threshold, text_metrics = find_best_threshold(y_val, text_val_prob)
    text_metrics["best_threshold"] = text_threshold
    logger.info("文本模型验证集: %s", text_metrics)

    text_weight, raw_recommended_threshold, raw_blend_val_metrics, blend_grid = find_best_blend(
        y_val, feature_val_prob, text_val_prob
    )
    blend_val_prob_raw = (1.0 - text_weight) * feature_val_prob + text_weight * text_val_prob
    calibrator, calibration_report = select_probability_calibrator(
        y_val,
        blend_val_prob_raw,
        min_samples=max(int(args.calibration_min_samples), 50),
        seed=args.seed,
    )
    blend_val_prob = apply_probability_calibrator(calibrator, blend_val_prob_raw)
    recommended_threshold, blend_val_metrics = find_best_threshold(y_val, blend_val_prob)

    short_text_threshold, short_text_metrics, short_count = compute_short_text_threshold(
        y_val,
        blend_val_prob,
        val_texts,
        short_text_length_upper=args.short_text_length_upper,
        target_recall=args.short_text_target_recall,
    )
    logger.info(
        "融合参数: text_weight=%.3f raw_threshold=%.3f calibrated_threshold=%.3f",
        text_weight,
        raw_recommended_threshold,
        recommended_threshold,
    )
    logger.info("融合验证集(原始概率): %s", raw_blend_val_metrics)
    logger.info("融合验证集(校准后概率): %s", blend_val_metrics)
    logger.info("校准报告: %s", calibration_report)
    logger.info(
        "短文本阈值: upper=%d threshold=%.3f short_count=%d metrics=%s",
        args.short_text_length_upper,
        short_text_threshold,
        short_count,
        short_text_metrics,
    )

    # 直接使用验证阶段选出的模型作为线上模型，避免“阈值/校准器与部署模型不同源”
    serving_feature_model = feature_model
    serving_text_model = text_model

    test_feature_prob = serving_feature_model.predict_proba(X_test)[:, 1]
    test_text_prob = serving_text_model.predict_proba(test_texts)[:, 1]
    test_blend_prob_raw = (1.0 - text_weight) * test_feature_prob + text_weight * test_text_prob
    test_blend_prob = apply_probability_calibrator(calibrator, test_blend_prob_raw)

    test_feature_metrics = evaluate_binary(y_test, test_feature_prob, threshold=0.5)
    test_text_metrics = evaluate_binary(y_test, test_text_prob, threshold=text_threshold)
    test_blend_metrics_raw = evaluate_binary(y_test, test_blend_prob_raw, threshold=raw_recommended_threshold)
    test_blend_metrics = evaluate_binary(y_test, test_blend_prob, threshold=recommended_threshold)
    test_calibration = {
        "ece_before": expected_calibration_error(y_test, test_blend_prob_raw),
        "ece_after": expected_calibration_error(y_test, test_blend_prob),
        "brier_before": brier_score(y_test, test_blend_prob_raw),
        "brier_after": brier_score(y_test, test_blend_prob),
        "log_loss_before": log_loss_score(y_test, test_blend_prob_raw),
        "log_loss_after": log_loss_score(y_test, test_blend_prob),
    }

    logger.info("测试集 特征模型: %s", test_feature_metrics)
    logger.info("测试集 文本模型: %s", test_text_metrics)
    logger.info("测试集 融合模型(原始概率): %s", test_blend_metrics_raw)
    logger.info("测试集 融合模型(校准后概率): %s", test_blend_metrics)
    logger.info("测试集 校准误差: %s", test_calibration)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_iteration = _load_previous_iteration(output_path)
    archived_model_path = _archive_existing_model(output_path) if args.archive_previous else None

    metadata = {
        "version": "3.2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_mode": "fixed_single_model_iterative",
        "training_iteration": int(previous_iteration + 1),
        "previous_model_backup": archived_model_path,
        "ai_data_dir": str(Path(args.ai_data).resolve()),
        "human_data_dir": str(Path(args.human_data).resolve()),
        "model_name": feature_name,
        "expected_feature_count": len(extractor.feature_names),
        "feature_names": extractor.feature_names,
        "min_text_length": args.min_length,
        "recommended_threshold": float(recommended_threshold),
        "recommended_threshold_raw": float(raw_recommended_threshold),
        "short_text_threshold": float(short_text_threshold),
        "short_text_length_upper": int(args.short_text_length_upper),
        "default_blend_alpha": 1.0,
        "text_model_weight": float(text_weight),
        "augment_short_ratio": float(max(0.0, min(args.augment_short_ratio, 1.0))),
        "augment_sentence_drop_prob": float(max(0.0, min(args.augment_sentence_drop_prob, 1.0))),
        "train_augmented_count": int(aug_count),
        "calibration_min_samples": int(max(int(args.calibration_min_samples), 50)),
        "sample_count": int(len(labels)),
        "class_distribution": {"ai": int(np.sum(labels == 1)), "human": int(np.sum(labels == 0))},
        "serving_model_source": "validation_selected_without_refit",
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "validation_metrics": {
            "feature_candidates": feature_val_metrics,
            "text_model": text_metrics,
            "blend_best_raw": raw_blend_val_metrics,
            "blend_best_calibrated": blend_val_metrics,
            "short_text": {"count": short_count, "metrics": short_text_metrics, "threshold": short_text_threshold},
            "blend_grid": blend_grid,
        },
        "test_metrics": {
            "feature_model": test_feature_metrics,
            "text_model": test_text_metrics,
            "blend_model_raw": test_blend_metrics_raw,
            "blend_model_calibrated": test_blend_metrics,
            "calibration": test_calibration,
        },
        "calibration": calibration_report,
    }

    bundle = {
        "classifier": serving_feature_model,  # 兼容旧读取逻辑
        "feature_classifier": serving_feature_model,
        "text_classifier": serving_text_model,
        "probability_calibrator": calibrator,
        "metadata": metadata,
    }
    joblib.dump(bundle, output_path)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if archived_model_path:
        logger.info("检测到旧模型，已归档备份: %s", archived_model_path)
    logger.info("训练迭代序号: %d", int(previous_iteration + 1))
    logger.info("模型已保存: %s", output_path)
    logger.info("元数据已保存: %s", metadata_path)


if __name__ == "__main__":
    main()
