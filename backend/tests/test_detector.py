import pytest
import types

import numpy as np

from backend.models.detector import AIGCDetector


class _FakeTransformerScore:
    def __init__(self, probability: float, confidence: float, chunk_count: int) -> None:
        self.probability = probability
        self.confidence = confidence
        self.chunk_count = chunk_count


class _FakeTransformerScorer:
    def __init__(self, probability: float = 0.8, available: bool = True) -> None:
        self.available = available
        self.model_id = "fake/transformer"
        self.reason = "ok" if available else "disabled_for_test"
        self._probability = probability

    def score_text(self, _text: str):
        if not self.available:
            return None
        return _FakeTransformerScore(probability=self._probability, confidence=0.9, chunk_count=1)


def test_detector_uses_heuristic_mode_when_model_missing():
    detector = AIGCDetector(model_path="/tmp/not_exists.joblib", min_text_length=20)
    assert detector.model_loaded is False
    assert detector.model_mode == "heuristic_only"


def test_detector_rejects_short_text():
    detector = AIGCDetector(model_path="/tmp/not_exists.joblib", min_text_length=50)
    with pytest.raises(ValueError):
        detector.analyze("太短了")


def test_detector_returns_expected_fields():
    detector = AIGCDetector(model_path="/tmp/not_exists.joblib", min_text_length=20)
    result = detector.analyze("这是一个用于测试的中文段落。" * 10, include_details=True)
    assert "aigc_score" in result
    assert "confidence" in result
    assert "features" in result
    assert "warnings" in result
    assert "details" in result
    assert "feature_contributions" in result["details"]
    assert isinstance(result["details"]["feature_contributions"], list)
    assert len(result["details"]["feature_contributions"]) > 0
    assert "calibration_enabled" in result["details"]
    assert "calibration_method" in result["details"]
    assert "calibration_metrics" in result["details"]


def test_detector_label_matches_score_threshold():
    detector = AIGCDetector(model_path="/tmp/not_exists.joblib", min_text_length=20)
    result = detector.analyze("这是一个用于一致性验证的中文文本。" * 10, include_details=False)
    score = float(result["aigc_score"])
    threshold = float(result["score_threshold"])
    expected = "ai" if score >= threshold else "human"
    assert result["label"] == expected


def test_detector_transformer_branch_can_drive_prediction():
    detector = AIGCDetector(
        model_path="/tmp/not_exists.joblib",
        min_text_length=20,
        transformer_scorer=_FakeTransformerScorer(probability=0.88, available=True),
    )
    result = detector.analyze("这是一个用于测试Transformer分支驱动预测的样本文本。" * 8, include_details=True)
    assert result["label"] == "ai"
    assert result["details"]["transformer_enabled"] is True
    assert result["details"]["transformer_probability"] is not None


def test_detector_transformer_unavailable_adds_warning():
    detector = AIGCDetector(
        model_path="/tmp/not_exists.joblib",
        min_text_length=20,
        transformer_scorer=_FakeTransformerScorer(probability=0.5, available=False),
    )
    result = detector.analyze("这是一个用于测试Transformer未启用提示信息的样本文本。" * 8, include_details=True)
    joined_warnings = " ".join(result.get("warnings", []))
    assert "Transformer 分支未生效" in joined_warnings


def test_detector_transformer_non_zh_route_skips_cjk_text(monkeypatch):
    monkeypatch.setenv("AIGC_TRANSFORMER_APPLY_TO", "non_zh")
    monkeypatch.setenv("AIGC_TRANSFORMER_CJK_THRESHOLD", "0.2")
    detector = AIGCDetector(
        model_path="/tmp/not_exists.joblib",
        min_text_length=20,
        transformer_scorer=_FakeTransformerScorer(probability=0.95, available=True),
    )
    result = detector.analyze("这是一个主要由中文字符组成的测试文本。" * 10, include_details=True)
    assert result["details"]["transformer_probability"] is None


def test_detector_can_use_non_zh_threshold(monkeypatch):
    monkeypatch.setenv("AIGC_NON_ZH_SCORE_THRESHOLD", "0.2")
    monkeypatch.setenv("AIGC_TRANSFORMER_CJK_THRESHOLD", "0.2")
    detector = AIGCDetector(
        model_path="/tmp/not_exists.joblib",
        min_text_length=20,
        transformer_scorer=_FakeTransformerScorer(probability=0.31, available=True),
    )
    result = detector.analyze("This is an English test sample with enough length for threshold routing." * 4, include_details=True)
    assert float(result["score_threshold"]) == 20.0


def test_detector_score_clip_avoids_absolute_extremes(monkeypatch):
    monkeypatch.setenv("AIGC_TRANSFORMER_WEIGHT", "1.0")
    monkeypatch.setenv("AIGC_SCORE_CLIP_EPS", "0.02")
    detector = AIGCDetector(
        model_path="/tmp/not_exists.joblib",
        min_text_length=20,
        transformer_scorer=_FakeTransformerScorer(probability=1.0, available=True),
    )
    result = detector.analyze("This is a long enough English sample used to test score clipping behavior." * 6, include_details=True)
    assert float(result["aigc_score"]) == 98.0
    assert float(result["details"]["aigc_probability_raw"]) == 1.0


def test_detector_contributions_include_transformer_branch(monkeypatch):
    monkeypatch.setenv("AIGC_TRANSFORMER_WEIGHT", "1.0")
    detector = AIGCDetector(
        model_path="/tmp/not_exists.joblib",
        min_text_length=20,
        transformer_scorer=_FakeTransformerScorer(probability=0.9, available=True),
    )
    result = detector.analyze("This is another long English sample for checking contribution rows." * 6, include_details=True)
    rows = result["details"].get("feature_contributions", [])
    assert any(row.get("feature") == "transformer_branch" for row in rows)


def test_detector_explainability_backfill_when_transformer_weight_is_one(monkeypatch):
    monkeypatch.setenv("AIGC_TRANSFORMER_WEIGHT", "1.0")
    monkeypatch.setenv("AIGC_EXPLAIN_CLASSIC_MIN_WEIGHT", "0.2")
    detector = AIGCDetector(
        model_path="/tmp/not_exists.joblib",
        min_text_length=20,
        transformer_scorer=_FakeTransformerScorer(probability=0.62, available=True),
    )

    def fake_combined(self, _text, _feature_vector):
        # 模拟“经典分支 + Transformer 分支”同时存在，且线上权重完全偏向 Transformer
        return 0.62, {
            "feature_probability": 0.78,
            "text_probability": 0.72,
            "classic_probability_raw": 0.76,
            "classic_probability": 0.76,
            "classic_probability_calibrated": False,
            "transformer_probability": 0.62,
            "transformer_confidence": 0.9,
            "transformer_chunk_count": 1,
            "transformer_weight_used": 1.0,
            "transformer_apply_to": "all",
            "transformer_cjk_ratio": 0.9,
        }

    def fake_local_signal(self, feature_vector):
        return np.array(feature_vector, dtype=float) - 0.5

    detector._combined_model_probability = types.MethodType(fake_combined, detector)
    detector._feature_model_local_signal = types.MethodType(fake_local_signal, detector)

    result = detector.analyze("这是一个用于解释性贡献回补测试的中文文本。" * 10, include_details=True)
    rows = result["details"].get("feature_contributions", [])
    non_transformer_rows = [row for row in rows if row.get("feature") != "transformer_branch"]
    assert any(abs(float(row.get("total_contribution", 0.0))) > 1e-6 for row in non_transformer_rows)
    assert any("解释性回补" in warning for warning in result.get("warnings", []))
    assert float(result["details"]["classic_weight_for_contribution"]) >= 0.19
