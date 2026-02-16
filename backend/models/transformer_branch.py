import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from backend.utils.text_processing import split_text_into_chunks
except ImportError:  # 兼容在backend目录执行
    from utils.text_processing import split_text_into_chunks


logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TransformerScore:
    probability: float
    confidence: float
    chunk_count: int


class TransformerAIGCScorer:
    """可选 Transformer AIGC 概率分支（支持任意二分类 sequence-classification 模型）。"""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_length: int = 512,
        chunk_size_chars: int = 600,
    ) -> None:
        self.model_id = model_id
        self.max_length = max(64, int(max_length))
        self.chunk_size_chars = max(200, int(chunk_size_chars))
        self.device = device
        self.available = False
        self.reason = ""

        self._tokenizer = None
        self._model = None
        self._torch = None
        self._ai_label_index = 1

        self._load()

    @classmethod
    def from_env(cls) -> Optional["TransformerAIGCScorer"]:
        enabled = _env_enabled("AIGC_ENABLE_TRANSFORMER", "false")
        model_id = os.getenv("AIGC_TRANSFORMER_MODEL_ID", "").strip()
        if not enabled:
            return None
        if not model_id:
            logger.warning("AIGC_ENABLE_TRANSFORMER 已启用，但未设置 AIGC_TRANSFORMER_MODEL_ID，跳过 Transformer 分支")
            return None
        device = os.getenv("AIGC_TRANSFORMER_DEVICE", "auto").strip()
        max_length_raw = os.getenv("AIGC_TRANSFORMER_MAX_LENGTH", "512")
        chunk_size_raw = os.getenv("AIGC_TRANSFORMER_CHUNK_SIZE_CHARS", "600")
        try:
            max_length = max(64, int(max_length_raw))
        except ValueError:
            max_length = 512
        try:
            chunk_size = max(200, int(chunk_size_raw))
        except ValueError:
            chunk_size = 600
        return cls(model_id=model_id, device=device, max_length=max_length, chunk_size_chars=chunk_size)

    def _resolve_device(self, torch_module) -> str:
        if self.device and self.device != "auto":
            return self.device
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _infer_ai_label_index(id2label: Dict[int, str], num_labels: int) -> int:
        if not id2label:
            return 1 if num_labels > 1 else 0
        normalized: Dict[int, str] = {}
        for idx, label in id2label.items():
            normalized[int(idx)] = str(label).strip().lower()

        ai_patterns = (
            r"\bai\b",
            r"gpt",
            r"chatgpt",
            r"machine",
            r"generated",
            r"fake",
            r"bot",
            r"positive",
            r"label_1",
            r"class_1",
        )
        for idx, label in normalized.items():
            if any(re.search(pattern, label) for pattern in ai_patterns):
                return idx
        if num_labels >= 2:
            return 1
        return 0

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            self.available = False
            self.reason = f"missing_dependency: {exc}"
            logger.warning("Transformer 分支依赖缺失，跳过加载: %s", exc)
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            resolved_device = self._resolve_device(torch)
            model = model.to(resolved_device)
            model.eval()

            config = getattr(model, "config", None)
            num_labels = int(getattr(config, "num_labels", 2))
            id2label = getattr(config, "id2label", {}) or {}
            self._ai_label_index = self._infer_ai_label_index(id2label, num_labels)

            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            self.device = resolved_device
            self.available = True
            self.reason = "ok"
            logger.info(
                "Transformer 分支加载成功: model=%s device=%s ai_label_index=%s",
                self.model_id,
                self.device,
                self._ai_label_index,
            )
        except Exception as exc:
            self.available = False
            self.reason = f"load_failed: {exc}"
            logger.warning("Transformer 分支加载失败，跳过: %s", exc)

    @staticmethod
    def _clamp(probability: float) -> float:
        return float(min(max(probability, 0.0), 1.0))

    def score_text(self, text: str) -> Optional[TransformerScore]:
        if not self.available or self._model is None or self._tokenizer is None or self._torch is None:
            return None
        if not text or not text.strip():
            return None

        chunks = split_text_into_chunks(text, max_chunk_size=self.chunk_size_chars, min_chunk_size=80)
        if not chunks:
            return None

        probs: List[float] = []
        confidences: List[float] = []
        lengths: List[int] = []

        with self._torch.no_grad():
            for chunk in chunks:
                encoded = self._tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                logits = outputs.logits.detach().cpu().numpy()[0]
                if logits.ndim != 1:
                    continue
                if logits.shape[0] <= self._ai_label_index:
                    continue
                exp = np.exp(logits - np.max(logits))
                softmax = exp / np.sum(exp)
                ai_prob = float(softmax[self._ai_label_index])
                top = float(np.max(softmax))
                second = float(np.partition(softmax, -2)[-2]) if softmax.shape[0] > 1 else 0.0
                margin = max(top - second, 0.0)
                probs.append(self._clamp(ai_prob))
                confidences.append(self._clamp(0.45 + margin * 1.2))
                lengths.append(max(len(chunk), 1))

        if not probs:
            return None

        weights = np.array(lengths, dtype=float)
        probability = float(np.average(np.array(probs, dtype=float), weights=weights))
        confidence = float(np.average(np.array(confidences, dtype=float), weights=weights))
        return TransformerScore(
            probability=self._clamp(probability),
            confidence=self._clamp(confidence),
            chunk_count=len(probs),
        )
