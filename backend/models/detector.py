import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

try:
    from backend.models.features import FeatureExtractor
    from backend.models.transformer_branch import TransformerAIGCScorer
    from backend.utils.text_processing import preprocess_text, split_text_into_chunks
except ImportError:  # 兼容从backend目录直接运行脚本
    from models.features import FeatureExtractor
    from models.transformer_branch import TransformerAIGCScorer
    from utils.text_processing import preprocess_text, split_text_into_chunks


logger = logging.getLogger(__name__)


def _default_model_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(repo_root, "data", "models")
    preferred = os.path.join(models_dir, "aigc_detector_model.joblib")
    # 固定单模型路径，不再自动切换多版本文件。
    return preferred


class AIGCDetector:
    """AIGC文本检测器（可训练模型 + 启发式兜底）。"""

    HEURISTIC_WEIGHTS: Dict[str, float] = {
        "char_entropy_norm": -0.22,
        "sentence_length_cv_norm": -0.18,
        "lexical_diversity": -0.24,
        "hapax_ratio": -0.16,
        "repetition_ratio": 0.20,
        "bigram_repetition_ratio": 0.18,
        "function_word_ratio": 0.08,
        "punctuation_ratio": -0.06,
        "long_word_ratio": -0.06,
        "pos_diversity": -0.12,
        "noun_verb_balance": 0.04,
        "avg_sentence_length_norm": 0.02,
    }
    FEATURE_DIRECTIONS: Dict[str, str] = {
        "char_entropy_norm": "low",
        "avg_sentence_length_norm": "varies",
        "sentence_length_cv_norm": "low",
        "lexical_diversity": "low",
        "hapax_ratio": "low",
        "repetition_ratio": "high",
        "bigram_repetition_ratio": "high",
        "function_word_ratio": "varies",
        "punctuation_ratio": "varies",
        "long_word_ratio": "low",
        "pos_diversity": "low",
        "noun_verb_balance": "varies",
    }
    ASSISTANT_STYLE_PATTERNS: Tuple[str, ...] = (
        r"首先",
        r"其次",
        r"最后",
        r"总之",
        r"第一",
        r"第二",
        r"第三",
        r"步骤",
        r"方案",
        r"建议",
        r"你可以",
        r"可以先",
        r"需要注意",
        r"从.{0,6}角度",
        r"简单说",
        r"具体来说",
        r"拆解",
        r"复盘",
        r"你这个",
        r"这个感受",
        r"如果你愿意",
        r"我们可以",
        r"你会发现",
        r"很多时候",
        r"往往",
        r"恰恰",
        r"现阶段",
        r"本质上",
        r"换句话说",
        r"可以做到",
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_text_length: int = 50,
        transformer_scorer: Optional[TransformerAIGCScorer] = None,
    ) -> None:
        self.feature_extractor = FeatureExtractor()
        self.min_text_length = min_text_length
        self.model_path = model_path or os.getenv("AIGC_MODEL_PATH", _default_model_path())

        self.model_loaded = False
        self.model = None
        self.feature_model = None
        self.text_model = None
        self.probability_calibrator = None
        self.model_metadata: Dict[str, object] = {}
        self.model_mode = "heuristic_only"
        self.model_version = "none"
        self.model_blend_alpha = 1.0
        self.calibration_enabled = os.getenv("AIGC_DISABLE_CALIBRATION", "").strip().lower() not in {"1", "true", "yes"}
        self.calibration_method = "none"
        self.calibration_metrics: Dict[str, float] = {}
        self.assistant_cue_blend = self._resolve_assistant_cue_blend()
        self.assistant_cue_threshold_shift = self._resolve_assistant_cue_threshold_shift()
        self.score_threshold = 0.5
        self.text_model_weight = 0.0
        self.transformer_weight = self._resolve_transformer_weight()
        self.transformer_apply_to = self._resolve_transformer_apply_to()
        self.transformer_cjk_threshold = self._resolve_transformer_cjk_threshold()
        self.non_zh_score_threshold = self._resolve_non_zh_score_threshold()
        self.short_text_threshold = 0.5
        self.short_text_length_upper = 160
        self.score_clip_eps = self._resolve_score_clip_eps()
        self.explain_classic_min_weight = self._resolve_explain_classic_min_weight()
        self.transformer_scorer = transformer_scorer if transformer_scorer is not None else TransformerAIGCScorer.from_env()
        self.transformer_enabled = self.transformer_scorer is not None and bool(self.transformer_scorer.available)
        self.transformer_model_id = self.transformer_scorer.model_id if self.transformer_scorer is not None else None

        self._load_model()
        if self.transformer_enabled:
            if self.model_mode == "ml_plus_heuristic":
                self.model_mode = "ml_transformer_plus_heuristic"
            elif self.model_mode == "heuristic_only":
                self.model_mode = "transformer_plus_heuristic"

    @classmethod
    def from_env(cls) -> "AIGCDetector":
        min_text_length_raw = os.getenv("AIGC_MIN_TEXT_LENGTH", "50")
        try:
            min_text_length = max(int(min_text_length_raw), 10)
        except ValueError:
            min_text_length = 50
        return cls(model_path=os.getenv("AIGC_MODEL_PATH"), min_text_length=min_text_length)

    def _load_model(self) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning("未找到模型文件，使用启发式模式: %s", self.model_path)
            return

        try:
            bundle = joblib.load(self.model_path)

            # 新格式
            if isinstance(bundle, dict) and "classifier" in bundle:
                metadata = bundle.get("metadata", {})
                expected_count = int(metadata.get("expected_feature_count", len(self.feature_extractor.feature_names)))
                if expected_count != len(self.feature_extractor.feature_names):
                    raise ValueError(
                        f"模型特征维度不匹配，期望 {expected_count}，当前 {len(self.feature_extractor.feature_names)}"
                    )

                self.model = bundle["classifier"]
                self.feature_model = bundle.get("feature_classifier", self.model)
                self.text_model = bundle.get("text_classifier")
                self.probability_calibrator = bundle.get("probability_calibrator")
                self.model_metadata = metadata
                self.model_version = str(metadata.get("version", "unknown"))
                self.model_blend_alpha = self._resolve_blend_alpha(metadata)
                self.score_threshold = self._resolve_score_threshold(metadata)
                self.text_model_weight = self._resolve_text_model_weight(metadata)
                self.short_text_threshold = self._resolve_short_text_threshold(metadata)
                self.short_text_length_upper = self._resolve_short_text_length_upper(metadata)
                calibration_meta = metadata.get("calibration", {})
                if isinstance(calibration_meta, dict):
                    self.calibration_method = str(calibration_meta.get("method", "none"))
                    self.calibration_metrics = {
                        "val_ece_before": float(calibration_meta.get("val_ece_before", 0.0)),
                        "val_ece_after": float(calibration_meta.get("val_ece_after", 0.0)),
                        "val_brier_before": float(calibration_meta.get("val_brier_before", 0.0)),
                        "val_brier_after": float(calibration_meta.get("val_brier_after", 0.0)),
                        "val_log_loss_before": float(calibration_meta.get("val_log_loss_before", 0.0)),
                        "val_log_loss_after": float(calibration_meta.get("val_log_loss_after", 0.0)),
                    }
                if isinstance(self.probability_calibrator, dict):
                    self.calibration_method = str(self.probability_calibrator.get("method", self.calibration_method))
                elif self.probability_calibrator is not None and self.calibration_method == "none":
                    self.calibration_method = "isotonic_regression"
                self.model_loaded = True
                self.model_mode = "ml_plus_heuristic"
                logger.info("成功加载模型: %s (version=%s)", self.model_path, self.model_version)
                return

            # 兼容旧格式
            if isinstance(bundle, dict):
                legacy_keys = [k for k in bundle.keys() if str(k).startswith("model_")]
                if legacy_keys:
                    self.model = bundle[legacy_keys[0]]
                    self.feature_model = self.model
                    self.text_model = None
                    self.probability_calibrator = None
                    self.model_loaded = True
                    self.model_mode = "legacy_ml_plus_heuristic"
                    self.model_version = "legacy"
                    self.model_blend_alpha = self._resolve_blend_alpha({})
                    self.score_threshold = self._resolve_score_threshold({})
                    self.text_model_weight = self._resolve_text_model_weight({})
                    self.short_text_threshold = self._resolve_short_text_threshold({})
                    self.short_text_length_upper = self._resolve_short_text_length_upper({})
                    logger.warning("加载旧版模型成功: %s", legacy_keys[0])
                    return

            raise ValueError("模型文件格式不受支持")
        except Exception as exc:
            logger.error("模型加载失败，回退启发式模式: %s", exc)
            self.model_loaded = False
            self.model = None
            self.feature_model = None
            self.text_model = None
            self.probability_calibrator = None
            self.model_mode = "heuristic_only"

    def _resolve_blend_alpha(self, metadata: Dict[str, object]) -> float:
        raw = os.getenv("AIGC_MODEL_BLEND_ALPHA")
        if raw is None:
            raw = str(metadata.get("default_blend_alpha", 1.0))
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return 1.0

    def _resolve_score_threshold(self, metadata: Dict[str, object]) -> float:
        raw = os.getenv("AIGC_SCORE_THRESHOLD")
        if raw is None:
            raw = str(metadata.get("recommended_threshold", 0.5))
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return 0.5

    def _resolve_text_model_weight(self, metadata: Dict[str, object]) -> float:
        raw = os.getenv("AIGC_TEXT_MODEL_WEIGHT")
        if raw is None:
            raw = str(metadata.get("text_model_weight", 0.0))
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return 0.0

    def _resolve_transformer_weight(self) -> float:
        raw = os.getenv("AIGC_TRANSFORMER_WEIGHT", "1.0")
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return 1.0

    def _resolve_transformer_apply_to(self) -> str:
        raw = os.getenv("AIGC_TRANSFORMER_APPLY_TO", "all").strip().lower()
        if raw in {"all", "zh", "non_zh"}:
            return raw
        return "all"

    def _resolve_transformer_cjk_threshold(self) -> float:
        raw = os.getenv("AIGC_TRANSFORMER_CJK_THRESHOLD", "0.2")
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return 0.2

    def _resolve_non_zh_score_threshold(self) -> Optional[float]:
        raw = os.getenv("AIGC_NON_ZH_SCORE_THRESHOLD")
        if raw is None or not raw.strip():
            return None
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return None

    def _resolve_short_text_threshold(self, metadata: Dict[str, object]) -> float:
        raw = os.getenv("AIGC_SHORT_TEXT_THRESHOLD")
        if raw is None:
            raw = str(metadata.get("short_text_threshold", metadata.get("recommended_threshold", 0.5)))
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return self.score_threshold

    def _resolve_short_text_length_upper(self, metadata: Dict[str, object]) -> int:
        raw = os.getenv("AIGC_SHORT_TEXT_LENGTH_UPPER")
        if raw is None:
            raw = str(metadata.get("short_text_length_upper", 160))
        try:
            return max(int(raw), 60)
        except ValueError:
            return 160

    def _resolve_score_clip_eps(self) -> float:
        raw = os.getenv("AIGC_SCORE_CLIP_EPS", "0.005")
        try:
            return float(min(max(float(raw), 0.0), 0.1))
        except ValueError:
            return 0.005

    def _resolve_explain_classic_min_weight(self) -> float:
        raw = os.getenv("AIGC_EXPLAIN_CLASSIC_MIN_WEIGHT", "0.18")
        try:
            value = float(raw)
        except ValueError:
            value = 0.18
        return float(min(max(value, 0.0), 0.5))

    def _resolve_assistant_cue_blend(self) -> float:
        raw = os.getenv("AIGC_ASSISTANT_CUE_BLEND", "0.28")
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return 0.28

    def _resolve_assistant_cue_threshold_shift(self) -> float:
        raw = os.getenv("AIGC_ASSISTANT_CUE_THRESHOLD_SHIFT", "0.22")
        try:
            return self._clamp_probability(float(raw))
        except ValueError:
            return 0.22

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return float(min(max(value, 0.0), 1.0))

    def _apply_score_clipping(self, value: float) -> float:
        clipped = self._clamp_probability(value)
        eps = float(self.score_clip_eps)
        if eps <= 0:
            return clipped
        return float(min(max(clipped, eps), 1.0 - eps))

    @staticmethod
    def _safe_logit(probability: float) -> float:
        p = float(min(max(probability, 1e-6), 1.0 - 1e-6))
        return math.log(p / (1.0 - p))

    @staticmethod
    def _estimate_cjk_ratio(text: str) -> float:
        if not text:
            return 0.0
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return 0.0
        cjk = sum(1 for c in chars if "\u4e00" <= c <= "\u9fff")
        return float(cjk / max(len(chars), 1))

    def _assistant_style_cue_probability(self, text: str) -> float:
        if not text:
            return 0.0
        normalized = text.strip()
        if len(normalized) < 40 or len(normalized) > 1200:
            return 0.0

        marker_count = 0
        for pattern in self.ASSISTANT_STYLE_PATTERNS:
            marker_count += len(re.findall(pattern, normalized))

        structure_count = len(re.findall(r"(第一|第二|第三|首先|其次|最后|总之)", normalized))
        directive_count = len(re.findall(r"(建议|可以|先|再|然后|步骤|方案|目标|拆解|复盘|需要|我们可以|你可以|你会发现)", normalized))
        discourse_count = len(re.findall(r"(其实|确实|通常|往往|恰恰|本质上|换句话说|另一方面|从这个角度)", normalized))
        second_person_count = len(re.findall(r"(你|你的|你可以)", normalized))
        empathy_opening = bool(re.match(r"^(你这个|这个感受|你说得|你提到|确实|是的)", normalized))

        cue = 0.0
        cue += min(marker_count / 7.0, 1.0) * 0.35
        cue += min(structure_count / 3.0, 1.0) * 0.20
        cue += min(directive_count / 8.0, 1.0) * 0.20
        cue += min(discourse_count / 8.0, 1.0) * 0.15
        cue += min(second_person_count / 6.0, 1.0) * 0.10
        if empathy_opening:
            cue += 0.10
        return self._clamp_probability(cue)

    def _heuristic_score(self, features: Dict[str, float]) -> float:
        # 正值倾向AIGC，负值倾向人工写作
        score = 0.5
        for name, weight in self.HEURISTIC_WEIGHTS.items():
            value = float(features.get(name, 0.5))
            score += (value - 0.5) * 2.0 * weight
        return self._clamp_probability(score)

    def _heuristic_contribution_terms(self, feature_dict: Dict[str, float]) -> np.ndarray:
        terms: List[float] = []
        for feature_name in self.feature_extractor.feature_names:
            weight = float(self.HEURISTIC_WEIGHTS.get(feature_name, 0.0))
            value = float(feature_dict.get(feature_name, 0.5))
            terms.append((value - 0.5) * 2.0 * weight)
        return np.array(terms, dtype=float)

    def _feature_direction_vector(self) -> np.ndarray:
        signs: List[float] = []
        for feature_name in self.feature_extractor.feature_names:
            direction = self.FEATURE_DIRECTIONS.get(feature_name, "varies")
            if direction == "high":
                signs.append(1.0)
            elif direction == "low":
                signs.append(-1.0)
            else:
                signs.append(0.35)
        return np.array(signs, dtype=float)

    @staticmethod
    def _rescale_terms(terms: np.ndarray, target_sum: float) -> np.ndarray:
        if terms.size == 0 or abs(target_sum) < 1e-12:
            return np.zeros_like(terms, dtype=float)

        current_sum = float(np.sum(terms))
        if abs(current_sum) > 1e-10:
            return terms * (target_sum / current_sum)

        abs_sum = float(np.sum(np.abs(terms)))
        if abs_sum <= 1e-10:
            return np.zeros_like(terms, dtype=float)

        direction = 1.0 if target_sum >= 0 else -1.0
        return direction * (np.abs(terms) / abs_sum) * abs(target_sum)

    def _feature_model_local_signal(self, feature_vector: List[float]) -> Optional[np.ndarray]:
        if not self.model_loaded or self.feature_model is None:
            return None

        x = np.array(feature_vector, dtype=float).reshape(1, -1)
        direction_vec = self._feature_direction_vector()
        try:
            estimator = self.feature_model
            transformed = x

            if hasattr(estimator, "steps") and estimator.steps:
                # 取pipeline最后一步作为分类器，其前面的步骤用于特征变换
                try:
                    transformed = estimator[:-1].transform(x)
                except Exception:
                    transformed = x
                estimator = estimator.steps[-1][1]

            if hasattr(estimator, "coef_"):
                coefs = np.array(getattr(estimator, "coef_"), dtype=float)
                if coefs.ndim == 2 and coefs.shape[0] >= 1:
                    return coefs[0] * transformed[0]

            if hasattr(estimator, "feature_importances_"):
                importances = np.array(getattr(estimator, "feature_importances_"), dtype=float)
                centered = (x[0] - 0.5) * 2.0
                if importances.shape[0] == centered.shape[0]:
                    return centered * importances * direction_vec
        except Exception as exc:
            logger.debug("特征贡献信号计算失败: %s", exc)

        return None

    def _build_feature_contributions(
        self,
        chunk_results: List[Dict[str, object]],
        weights: np.ndarray,
    ) -> List[Dict[str, object]]:
        feature_names = list(self.feature_extractor.feature_names)
        feature_count = len(feature_names)
        if feature_count <= 0:
            return []

        total_contrib = np.average(
            np.array([item["feature_contributions"] for item in chunk_results], dtype=float), axis=0, weights=weights
        )
        heuristic_contrib = np.average(
            np.array([item["heuristic_feature_contributions"] for item in chunk_results], dtype=float),
            axis=0,
            weights=weights,
        )
        model_contrib = np.average(
            np.array([item["model_feature_contributions"] for item in chunk_results], dtype=float), axis=0, weights=weights
        )
        assistant_cue_probability = float(
            np.average(np.array([float(item.get("assistant_cue_probability", 0.0)) for item in chunk_results], dtype=float), weights=weights)
        )
        assistant_cue_contribution = float(
            np.average(np.array([float(item.get("assistant_cue_contribution", 0.0)) for item in chunk_results], dtype=float), weights=weights)
        )
        transformer_branch_contribution = float(
            np.average(np.array([float(item.get("transformer_branch_contribution", 0.0)) for item in chunk_results], dtype=float), weights=weights)
        )
        transformer_active = np.array(
            [1.0 if item.get("transformer_probability") is not None else 0.0 for item in chunk_results], dtype=float
        )
        transformer_active_ratio = float(np.average(transformer_active, weights=weights)) if weights.size > 0 else 0.0
        transformer_probability = float(
            np.average(
                np.array(
                    [
                        float(item.get("transformer_probability", 0.0))
                        if item.get("transformer_probability") is not None
                        else 0.0
                        for item in chunk_results
                    ],
                    dtype=float,
                ),
                weights=weights,
            )
        )
        aggregated_features = np.average(
            np.array(
                [
                    [float(item["features"].get(feature_name, 0.5)) for feature_name in feature_names]
                    for item in chunk_results
                ],
                dtype=float,
            ),
            axis=0,
            weights=weights,
        )

        rows: List[Dict[str, object]] = []
        for idx, feature_name in enumerate(feature_names):
            total = float(total_contrib[idx])
            heuristic = float(heuristic_contrib[idx])
            model = float(model_contrib[idx])
            value = float(aggregated_features[idx])

            if total > 0.002:
                impact = "increase_ai_risk"
            elif total < -0.002:
                impact = "decrease_ai_risk"
            else:
                impact = "neutral"

            rows.append(
                {
                    "feature": feature_name,
                    "direction": self.FEATURE_DIRECTIONS.get(feature_name, "varies"),
                    "value": round(value, 6),
                    "value_percent": round(value * 100.0, 2),
                    "total_contribution": round(total, 6),
                    "total_contribution_percent_points": round(total * 100.0, 2),
                    "heuristic_contribution": round(heuristic, 6),
                    "heuristic_contribution_percent_points": round(heuristic * 100.0, 2),
                    "model_contribution": round(model, 6),
                    "model_contribution_percent_points": round(model * 100.0, 2),
                    "impact": impact,
                }
            )

        rows.sort(key=lambda item: abs(float(item["total_contribution"])), reverse=True)
        if abs(assistant_cue_contribution) > 1e-6:
            rows.append(
                {
                    "feature": "assistant_style_cue",
                    "direction": "high",
                    "value": round(assistant_cue_probability, 6),
                    "value_percent": round(assistant_cue_probability * 100.0, 2),
                    "total_contribution": round(assistant_cue_contribution, 6),
                    "total_contribution_percent_points": round(assistant_cue_contribution * 100.0, 2),
                    "heuristic_contribution": 0.0,
                    "heuristic_contribution_percent_points": 0.0,
                    "model_contribution": 0.0,
                    "model_contribution_percent_points": 0.0,
                    "impact": "increase_ai_risk" if assistant_cue_contribution > 0 else "neutral",
                }
            )
            rows.sort(key=lambda item: abs(float(item["total_contribution"])), reverse=True)
        if abs(transformer_branch_contribution) > 1e-6 and transformer_active_ratio > 0:
            rows.append(
                {
                    "feature": "transformer_branch",
                    "direction": "high",
                    "value": round(transformer_probability, 6),
                    "value_percent": round(transformer_probability * 100.0, 2),
                    "total_contribution": round(transformer_branch_contribution, 6),
                    "total_contribution_percent_points": round(transformer_branch_contribution * 100.0, 2),
                    "heuristic_contribution": 0.0,
                    "heuristic_contribution_percent_points": 0.0,
                    "model_contribution": round(transformer_branch_contribution, 6),
                    "model_contribution_percent_points": round(transformer_branch_contribution * 100.0, 2),
                    "impact": "increase_ai_risk" if transformer_branch_contribution > 0 else "decrease_ai_risk",
                }
            )
            rows.sort(key=lambda item: abs(float(item["total_contribution"])), reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _feature_model_probability(self, feature_vector: List[float]) -> Optional[float]:
        if not self.model_loaded or self.feature_model is None:
            return None

        try:
            if hasattr(self.feature_model, "predict_proba"):
                return float(self.feature_model.predict_proba([feature_vector])[0][1])
            if hasattr(self.feature_model, "decision_function"):
                decision = float(self.feature_model.decision_function([feature_vector])[0])
                return self._sigmoid(decision)
            return None
        except Exception as exc:
            logger.error("特征模型预测失败: %s", exc)
            return None

    def _text_model_probability(self, text: str) -> Optional[float]:
        if not self.model_loaded or self.text_model is None:
            return None
        try:
            if hasattr(self.text_model, "predict_proba"):
                return float(self.text_model.predict_proba([text])[0][1])
            if hasattr(self.text_model, "decision_function"):
                decision = float(self.text_model.decision_function([text])[0])
                return self._sigmoid(decision)
            return None
        except Exception as exc:
            logger.error("文本模型预测失败: %s", exc)
            return None

    def _transformer_probability(self, text: str) -> Optional[Tuple[float, float, int, str, float]]:
        if not self.transformer_enabled or self.transformer_scorer is None:
            return None
        cjk_ratio = self._estimate_cjk_ratio(text)
        if self.transformer_apply_to == "zh" and cjk_ratio < self.transformer_cjk_threshold:
            return None
        if self.transformer_apply_to == "non_zh" and cjk_ratio >= self.transformer_cjk_threshold:
            return None
        try:
            result = self.transformer_scorer.score_text(text)
            if result is None:
                return None
            return (
                self._clamp_probability(float(result.probability)),
                self._clamp_probability(float(result.confidence)),
                int(result.chunk_count),
                self.transformer_apply_to,
                cjk_ratio,
            )
        except Exception as exc:
            logger.warning("Transformer 分支预测失败，忽略本次分支输出: %s", exc)
            return None

    def _combined_model_probability(self, text: str, feature_vector: List[float]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        feature_prob = self._feature_model_probability(feature_vector)
        text_prob = self._text_model_probability(text)
        transformer_pack = self._transformer_probability(text)
        transformer_prob = None if transformer_pack is None else float(transformer_pack[0])
        transformer_conf = None if transformer_pack is None else float(transformer_pack[1])
        transformer_chunk_count = None if transformer_pack is None else int(transformer_pack[2])
        transformer_apply_to = None if transformer_pack is None else str(transformer_pack[3])
        transformer_cjk_ratio = None if transformer_pack is None else float(transformer_pack[4])

        classic_prob_raw: Optional[float]
        if feature_prob is None and text_prob is None:
            classic_prob_raw = None
        elif feature_prob is None:
            classic_prob_raw = text_prob
        elif text_prob is None:
            classic_prob_raw = feature_prob
        else:
            w_text = self.text_model_weight
            classic_prob_raw = (1.0 - w_text) * feature_prob + w_text * text_prob
        classic_prob = self._calibrate_probability(classic_prob_raw) if classic_prob_raw is not None else None
        classic_calibrated = bool(
            classic_prob_raw is not None
            and classic_prob is not None
            and self.calibration_enabled
            and self.probability_calibrator is not None
        )

        if classic_prob is None and transformer_prob is None:
            return None, {
                "feature_probability": None,
                "text_probability": None,
                "classic_probability_raw": None,
                "classic_probability": None,
                "classic_probability_calibrated": False,
                "transformer_probability": None,
                "transformer_confidence": None,
                "transformer_chunk_count": None,
                "transformer_weight_used": None,
                "transformer_apply_to": None,
                "transformer_cjk_ratio": transformer_cjk_ratio,
            }
        if classic_prob is None:
            return transformer_prob, {
                "feature_probability": feature_prob,
                "text_probability": text_prob,
                "classic_probability_raw": None,
                "classic_probability": None,
                "classic_probability_calibrated": False,
                "transformer_probability": transformer_prob,
                "transformer_confidence": transformer_conf,
                "transformer_chunk_count": transformer_chunk_count,
                "transformer_weight_used": 1.0,
                "transformer_apply_to": transformer_apply_to,
                "transformer_cjk_ratio": transformer_cjk_ratio,
            }
        if transformer_prob is None:
            return classic_prob, {
                "feature_probability": feature_prob,
                "text_probability": text_prob,
                "classic_probability_raw": classic_prob_raw,
                "classic_probability": classic_prob,
                "classic_probability_calibrated": classic_calibrated,
                "transformer_probability": None,
                "transformer_confidence": None,
                "transformer_chunk_count": None,
                "transformer_weight_used": 0.0,
                "transformer_apply_to": self.transformer_apply_to if self.transformer_enabled else None,
                "transformer_cjk_ratio": transformer_cjk_ratio,
            }

        w_trans = self.transformer_weight
        combined = (1.0 - w_trans) * classic_prob + w_trans * transformer_prob
        return combined, {
            "feature_probability": feature_prob,
            "text_probability": text_prob,
            "classic_probability_raw": classic_prob_raw,
            "classic_probability": classic_prob,
            "classic_probability_calibrated": classic_calibrated,
            "transformer_probability": transformer_prob,
            "transformer_confidence": transformer_conf,
            "transformer_chunk_count": transformer_chunk_count,
            "transformer_weight_used": w_trans,
            "transformer_apply_to": transformer_apply_to,
            "transformer_cjk_ratio": transformer_cjk_ratio,
        }

    def _calibrate_probability(self, probability: Optional[float]) -> Optional[float]:
        if probability is None:
            return None

        prob = self._clamp_probability(float(probability))
        if not self.calibration_enabled:
            return prob
        if self.probability_calibrator is None:
            return prob
        try:
            calibrator = self.probability_calibrator
            if isinstance(calibrator, dict):
                method = str(calibrator.get("method", "none"))
                if method == "temperature_scaling":
                    temperature = max(float(calibrator.get("temperature", 1.0)), 0.05)
                    calibrated = self._sigmoid(self._safe_logit(prob) / temperature)
                    return self._clamp_probability(calibrated)
                if method == "isotonic_regression":
                    model = calibrator.get("model")
                    if model is not None and hasattr(model, "predict"):
                        calibrated = self._clamp_probability(float(model.predict([prob])[0]))
                        # 防止校准器把中等概率压成极端值（常见于过拟合的 isotonic）
                        if (prob >= 0.08 and calibrated <= 0.02) or (prob <= 0.92 and calibrated >= 0.98):
                            return self._clamp_probability(0.75 * prob + 0.25 * calibrated)
                        if abs(calibrated - prob) > 0.35:
                            return self._clamp_probability(0.6 * prob + 0.4 * calibrated)
                        return calibrated
                    return prob
                return prob

            if hasattr(calibrator, "predict"):
                calibrated = float(calibrator.predict([prob])[0])
                return self._clamp_probability(calibrated)
            return prob
        except Exception as exc:
            logger.debug("概率校准失败，回退未校准概率: %s", exc)
            return prob

    def _analyze_chunk(self, chunk: str) -> Dict[str, object]:
        feature_vector = self.feature_extractor.extract_features(chunk)
        feature_dict = self.feature_extractor.get_feature_dict(feature_vector)

        heuristic_prob = self._heuristic_score(feature_dict)
        model_prob_raw, model_prob_details = self._combined_model_probability(chunk, feature_vector)
        model_prob = model_prob_raw
        heuristic_terms = self._heuristic_contribution_terms(feature_dict)
        transformer_effect = 0.0
        classic_weight_for_contribution = 0.0
        transformer_weight_for_contribution = 0.0
        explainability_floor_applied = False

        if model_prob is None:
            final_prob = heuristic_prob
            # 启发式模式置信度上限较低，防止过度自信
            confidence = min(0.65, 0.45 + abs(final_prob - 0.5) * 0.4)
            method = "heuristic_only"
            heuristic_effect = final_prob - 0.5
            model_effect = 0.0
            heuristic_contrib = self._rescale_terms(heuristic_terms, heuristic_effect)
            model_contrib = np.zeros_like(heuristic_contrib)
            classic_weight_for_contribution = 0.0
            transformer_weight_for_contribution = 0.0
        else:
            alpha = self.model_blend_alpha
            final_prob = alpha * model_prob + (1.0 - alpha) * heuristic_prob
            confidence = min(0.95, 0.55 + abs(model_prob - 0.5) * 0.9)
            method = self.model_mode
            heuristic_effect = (1.0 - alpha) * (heuristic_prob - 0.5)
            model_effect = alpha * (model_prob - 0.5)
            heuristic_contrib = self._rescale_terms(heuristic_terms, heuristic_effect)

            model_signal = self._feature_model_local_signal(feature_vector)
            if model_signal is None or model_signal.shape[0] != len(heuristic_terms):
                model_signal = heuristic_terms.copy()
            if float(np.sum(np.abs(model_signal))) <= 1e-12:
                centered = (np.array(feature_vector, dtype=float) - 0.5) * 2.0
                model_signal = centered * self._feature_direction_vector()
            classic_prob = model_prob_details.get("classic_probability")
            transformer_prob = model_prob_details.get("transformer_probability")
            transformer_weight_used = model_prob_details.get("transformer_weight_used")
            if classic_prob is not None and transformer_prob is not None:
                w_trans = self._clamp_probability(
                    float(transformer_weight_used if transformer_weight_used is not None else self.transformer_weight)
                )
                raw_classic_weight = self._clamp_probability(1.0 - w_trans)
                classic_weight_for_contribution = raw_classic_weight
                if raw_classic_weight <= 1e-8 and abs(float(classic_prob) - 0.5) > 1e-8:
                    classic_weight_for_contribution = self.explain_classic_min_weight
                    explainability_floor_applied = classic_weight_for_contribution > 0.0
                transformer_weight_for_contribution = self._clamp_probability(1.0 - classic_weight_for_contribution)
                classic_effect = alpha * classic_weight_for_contribution * (float(classic_prob) - 0.5)
                transformer_effect = alpha * transformer_weight_for_contribution * (float(transformer_prob) - 0.5)
                residual = model_effect - (classic_effect + transformer_effect)
                transformer_effect += residual
            elif classic_prob is not None:
                classic_effect = model_effect
                classic_weight_for_contribution = 1.0
                transformer_weight_for_contribution = 0.0
            elif transformer_prob is not None:
                classic_effect = 0.0
                transformer_effect = model_effect
                classic_weight_for_contribution = 0.0
                transformer_weight_for_contribution = 1.0
            else:
                classic_effect = model_effect
                classic_weight_for_contribution = 1.0
                transformer_weight_for_contribution = 0.0
            model_contrib = self._rescale_terms(model_signal, classic_effect)

        pre_cue_prob = self._clamp_probability(final_prob)
        assistant_cue_prob = self._assistant_style_cue_probability(chunk)
        # assistant 线索用于阈值调节，不直接改写概率，避免破坏概率可校准性
        final_prob = pre_cue_prob
        assistant_cue_contribution = 0.0

        return {
            "probability": self._clamp_probability(final_prob),
            "confidence": self._clamp_probability(confidence),
            "features": feature_dict,
            "char_count": len(chunk),
            "method": method,
            "model_probability": None if model_prob is None else self._clamp_probability(model_prob),
            "model_probability_raw": None if model_prob_raw is None else self._clamp_probability(model_prob_raw),
            "calibration_applied": bool(model_prob_details.get("classic_probability_calibrated", False)),
            "calibration_skipped_due_to_transformer": False,
            "feature_probability": model_prob_details["feature_probability"],
            "text_probability": model_prob_details["text_probability"],
            "classic_probability_raw": model_prob_details.get("classic_probability_raw"),
            "classic_probability": model_prob_details.get("classic_probability"),
            "classic_probability_calibrated": model_prob_details.get("classic_probability_calibrated"),
            "transformer_probability": model_prob_details.get("transformer_probability"),
            "transformer_confidence": model_prob_details.get("transformer_confidence"),
            "transformer_chunk_count": model_prob_details.get("transformer_chunk_count"),
            "transformer_weight_used": model_prob_details.get("transformer_weight_used"),
            "transformer_apply_to": model_prob_details.get("transformer_apply_to"),
            "transformer_cjk_ratio": model_prob_details.get("transformer_cjk_ratio"),
            "transformer_branch_contribution": float(transformer_effect),
            "classic_weight_for_contribution": float(classic_weight_for_contribution),
            "transformer_weight_for_contribution": float(transformer_weight_for_contribution),
            "explainability_floor_applied": bool(explainability_floor_applied),
            "heuristic_probability": heuristic_prob,
            "feature_contributions": (heuristic_contrib + model_contrib).tolist(),
            "heuristic_feature_contributions": heuristic_contrib.tolist(),
            "model_feature_contributions": model_contrib.tolist(),
            "assistant_cue_probability": assistant_cue_prob,
            "assistant_cue_contribution": assistant_cue_contribution,
        }

    def analyze(self, text: str, include_details: bool = False) -> Dict[str, object]:
        if not isinstance(text, str):
            raise ValueError("text 字段必须是字符串")

        cleaned = preprocess_text(text, preserve_newlines=True)
        if len(cleaned) < self.min_text_length:
            raise ValueError(f"文本长度不足，当前为 {len(cleaned)}，至少需要 {self.min_text_length} 个字符")

        chunks = split_text_into_chunks(cleaned, max_chunk_size=900, min_chunk_size=120)
        if not chunks:
            raise ValueError("文本预处理后为空，无法检测")
        text_cjk_ratio = self._estimate_cjk_ratio(cleaned)

        chunk_results = [self._analyze_chunk(chunk) for chunk in chunks]
        weights = np.array([max(int(item["char_count"]), 1) for item in chunk_results], dtype=float)

        probabilities = np.array([float(item["probability"]) for item in chunk_results], dtype=float)
        confidences = np.array([float(item["confidence"]) for item in chunk_results], dtype=float)

        final_probability = float(np.average(probabilities, weights=weights))
        final_confidence = float(np.average(confidences, weights=weights))

        # 分段结果方差越大，降低整体置信度
        variance = float(np.var(probabilities))
        consistency_penalty = math.exp(-4.5 * variance)
        final_confidence = self._clamp_probability(final_confidence * consistency_penalty)

        aggregated_features: Dict[str, float] = {}
        for feature_name in self.feature_extractor.feature_names:
            values = np.array([float(item["features"].get(feature_name, 0.5)) for item in chunk_results], dtype=float)
            aggregated_features[feature_name] = float(np.average(values, weights=weights))

        warnings: List[str] = []
        if not self.model_loaded:
            warnings.append("当前未加载训练模型，结果来自启发式分析，可靠性较低。")
        if self.transformer_scorer is not None and not self.transformer_enabled:
            reason = getattr(self.transformer_scorer, "reason", "unknown")
            warnings.append(f"Transformer 分支未生效（{reason}），当前仅使用轻量模型分支。")
        explainability_floor_ratio = float(
            np.average(
                np.array([1.0 if bool(item.get("explainability_floor_applied", False)) else 0.0 for item in chunk_results], dtype=float),
                weights=weights,
            )
        )
        if explainability_floor_ratio > 0:
            warnings.append("当前为 Transformer 主导模式，关键影响因子中的传统特征贡献为解释性回补，不影响最终AIGC率。")
        effective_threshold = self.score_threshold
        if len(cleaned) < 120:
            warnings.append("文本较短，检测稳定性可能下降。建议输入120字以上文本。")
            final_confidence = self._clamp_probability(final_confidence * 0.85)
        if len(cleaned) <= self.short_text_length_upper:
            effective_threshold = self.short_text_threshold
        if self.non_zh_score_threshold is not None and text_cjk_ratio < self.transformer_cjk_threshold:
            effective_threshold = self.non_zh_score_threshold

        assistant_cue_avg = float(
            np.average(
                np.array([float(item.get("assistant_cue_probability", 0.0)) for item in chunk_results], dtype=float),
                weights=weights,
            )
        )
        transformer_prob_avg = float(
            np.average(
                np.array(
                    [
                        float(item.get("transformer_probability", 0.0))
                        if item.get("transformer_probability") is not None
                        else 0.0
                        for item in chunk_results
                    ],
                    dtype=float,
                ),
                weights=weights,
            )
        )
        transformer_cjk_ratio_avg = float(
            np.average(
                np.array(
                    [
                        float(item.get("transformer_cjk_ratio", 0.0))
                        if item.get("transformer_cjk_ratio") is not None
                        else 0.0
                        for item in chunk_results
                    ],
                    dtype=float,
                ),
                weights=weights,
            )
        )
        transformer_active_ratio = float(
            np.average(
                np.array(
                    [
                        1.0 if item.get("transformer_probability") is not None else 0.0
                        for item in chunk_results
                    ],
                    dtype=float,
                ),
                weights=weights,
            )
        )
        transformer_branch_contribution_avg = float(
            np.average(
                np.array([float(item.get("transformer_branch_contribution", 0.0)) for item in chunk_results], dtype=float),
                weights=weights,
            )
        )
        classic_weight_for_contribution_avg = float(
            np.average(
                np.array([float(item.get("classic_weight_for_contribution", 0.0)) for item in chunk_results], dtype=float),
                weights=weights,
            )
        )
        transformer_weight_for_contribution_avg = float(
            np.average(
                np.array([float(item.get("transformer_weight_for_contribution", 0.0)) for item in chunk_results], dtype=float),
                weights=weights,
            )
        )
        if self.assistant_cue_threshold_shift > 0 and assistant_cue_avg > 0:
            shift = self.assistant_cue_threshold_shift
            floor = 0.24
            if assistant_cue_avg >= 0.25 and final_probability <= 0.55:
                shift = min(0.32, shift + 0.08)
            if assistant_cue_avg >= 0.40 and len(cleaned) <= max(self.short_text_length_upper + 100, 260):
                shift = min(0.42, shift + 0.12)
                floor = 0.21
            effective_threshold = max(floor, effective_threshold - shift * assistant_cue_avg)

        if (
            len(cleaned) <= max(self.short_text_length_upper + 80, 220)
            and final_probability >= 0.23
            and assistant_cue_avg >= 0.18
            and effective_threshold > 0.23
        ):
            effective_threshold = 0.23
            warnings.append("文本包含明显对话式回答线索，已在边界区间适度提升AIGC检出敏感度。")

        final_probability_raw = self._clamp_probability(final_probability)
        final_probability = self._apply_score_clipping(final_probability_raw)
        ai_label = final_probability >= effective_threshold

        response: Dict[str, object] = {
            "aigc_score": round(self._clamp_probability(final_probability) * 100.0, 2),
            "confidence": round(self._clamp_probability(final_confidence) * 100.0, 2),
            "label": "ai" if ai_label else "human",
            "score_threshold": round(effective_threshold * 100.0, 2),
            "features": aggregated_features,
            "model_mode": self.model_mode,
            "warnings": warnings,
        }

        if include_details:
            feature_contributions = self._build_feature_contributions(chunk_results, weights)
            response["details"] = {
                "chunk_count": len(chunk_results),
                "chunk_probabilities": [round(float(p) * 100.0, 2) for p in probabilities],
                "model_version": self.model_version,
                "model_path": self.model_path if self.model_loaded else None,
                "blend_alpha": self.model_blend_alpha,
                "text_model_weight": self.text_model_weight,
                "transformer_enabled": bool(self.transformer_enabled),
                "transformer_model_id": self.transformer_model_id,
                "transformer_weight": self.transformer_weight,
                "transformer_apply_to": self.transformer_apply_to,
                "transformer_cjk_threshold": self.transformer_cjk_threshold,
                "text_cjk_ratio": text_cjk_ratio,
                "non_zh_score_threshold": self.non_zh_score_threshold,
                "transformer_cjk_ratio": transformer_cjk_ratio_avg,
                "transformer_probability": transformer_prob_avg if transformer_active_ratio > 0 else None,
                "transformer_active_ratio": round(transformer_active_ratio, 4),
                "transformer_branch_contribution_percent_points": round(transformer_branch_contribution_avg * 100.0, 2),
                "classic_weight_for_contribution": round(classic_weight_for_contribution_avg, 4),
                "transformer_weight_for_contribution": round(transformer_weight_for_contribution_avg, 4),
                "explain_classic_min_weight": self.explain_classic_min_weight,
                "explainability_floor_applied_ratio": round(explainability_floor_ratio, 4),
                "short_text_threshold": self.short_text_threshold,
                "short_text_length_upper": self.short_text_length_upper,
                "calibration_enabled": bool(self.calibration_enabled and self.probability_calibrator is not None),
                "calibration_method": self.calibration_method,
                "calibration_metrics": self.calibration_metrics,
                "assistant_cue_blend": self.assistant_cue_blend,
                "assistant_cue_threshold_shift": self.assistant_cue_threshold_shift,
                "assistant_cue_probability": assistant_cue_avg,
                "aigc_probability_raw": round(final_probability_raw, 6),
                "score_clip_eps": self.score_clip_eps,
                "feature_contributions": feature_contributions,
                "top_ai_risk_increasers": [row for row in feature_contributions if row["impact"] == "increase_ai_risk"][:5],
                "top_ai_risk_reducers": [row for row in feature_contributions if row["impact"] == "decrease_ai_risk"][:5],
            }

        return response
