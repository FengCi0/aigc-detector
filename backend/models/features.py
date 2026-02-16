import re
from collections import Counter
from typing import Dict, List

import jieba
import jieba.posseg as pseg
import numpy as np
from scipy.stats import entropy


class FeatureExtractor:
    """中文文本特征提取器（轻量、可解释、固定维度）。"""

    FEATURE_NAMES = [
        "char_entropy_norm",
        "avg_sentence_length_norm",
        "sentence_length_cv_norm",
        "lexical_diversity",
        "hapax_ratio",
        "repetition_ratio",
        "bigram_repetition_ratio",
        "function_word_ratio",
        "punctuation_ratio",
        "long_word_ratio",
        "pos_diversity",
        "noun_verb_balance",
    ]

    def __init__(self) -> None:
        self.feature_names = list(self.FEATURE_NAMES)
        self.function_words = {
            "的",
            "了",
            "是",
            "在",
            "和",
            "也",
            "就",
            "都",
            "而",
            "及",
            "与",
            "或",
            "并",
            "被",
            "将",
            "把",
            "对",
            "从",
            "到",
            "给",
            "于",
            "以",
            "为",
            "这",
            "那",
            "这些",
            "那些",
            "一个",
            "我们",
            "你",
            "我",
            "他",
            "她",
            "它",
        }

    def extract_features(self, text: str) -> List[float]:
        if not text or len(text.strip()) < 10:
            return [0.5] * len(self.feature_names)

        normalized = text.strip()
        sentences = [s.strip() for s in re.split(r"[。！？!?；;\n]+", normalized) if s.strip()]
        words = [w.strip() for w in jieba.cut(normalized) if w.strip()]
        tags = [flag for _, flag in pseg.cut(normalized)]
        chars = [c for c in normalized if not c.isspace()]

        if not words or not chars:
            return [0.5] * len(self.feature_names)

        sentence_lengths = [len(s) for s in sentences] or [len(normalized)]
        word_counter = Counter(words)
        unique_words = len(word_counter)
        total_words = len(words)

        features = [
            self._char_entropy(chars),
            self._avg_sentence_length(sentence_lengths),
            self._sentence_length_cv(sentence_lengths),
            self._safe_div(unique_words, total_words),
            self._safe_div(sum(1 for _, c in word_counter.items() if c == 1), total_words),
            1.0 - self._safe_div(unique_words, total_words),
            self._bigram_repetition(words),
            self._safe_div(sum(1 for w in words if w in self.function_words), total_words),
            self._punctuation_ratio(normalized, len(chars)),
            self._safe_div(sum(1 for w in words if len(w) >= 3), total_words),
            self._pos_diversity_from_tags(tags),
            self._noun_verb_balance_from_tags(tags),
        ]
        return [float(min(max(v, 0.0), 1.0)) for v in features]

    def get_feature_dict(self, feature_vector: List[float]) -> Dict[str, float]:
        return dict(zip(self.feature_names, feature_vector))

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        if b == 0:
            return 0.0
        return a / b

    def _char_entropy(self, chars: List[str]) -> float:
        freq = Counter(chars)
        probs = [count / len(chars) for count in freq.values()]
        raw = float(entropy(probs, base=2)) if probs else 0.0
        return min(raw / 7.0, 1.0)

    @staticmethod
    def _avg_sentence_length(sentence_lengths: List[int]) -> float:
        avg_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        return min(avg_len / 60.0, 1.0)

    @staticmethod
    def _sentence_length_cv(sentence_lengths: List[int]) -> float:
        mean_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        if mean_len <= 0:
            return 0.0
        cv = float(np.std(sentence_lengths) / mean_len)
        return min(cv / 2.0, 1.0)

    @staticmethod
    def _bigram_repetition(words: List[str]) -> float:
        if len(words) < 2:
            return 0.0
        bigrams = [f"{words[i]}|{words[i + 1]}" for i in range(len(words) - 1)]
        counts = Counter(bigrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return min(repeated / len(bigrams), 1.0)

    @staticmethod
    def _punctuation_ratio(text: str, total_chars: int) -> float:
        punct_count = len(re.findall(r"[，。！？；：、“”‘’《》【】（）\(\)\[\],.:;!?\"'\-—…]", text))
        if total_chars <= 0:
            return 0.0
        return min(punct_count / total_chars, 1.0)

    @staticmethod
    def _pos_diversity(text: str) -> float:
        tags = [flag for _, flag in pseg.cut(text)]
        return FeatureExtractor._pos_diversity_from_tags(tags)

    @staticmethod
    def _pos_diversity_from_tags(tags: List[str]) -> float:
        if not tags:
            return 0.0
        counts = Counter(tags)
        probs = [c / len(tags) for c in counts.values()]
        raw_entropy = float(entropy(probs, base=2))
        max_entropy = np.log2(len(counts) + 1)
        if max_entropy <= 0:
            return 0.0
        return min(raw_entropy / max_entropy, 1.0)

    @staticmethod
    def _noun_verb_balance(text: str) -> float:
        tags = [flag for _, flag in pseg.cut(text)]
        return FeatureExtractor._noun_verb_balance_from_tags(tags)

    @staticmethod
    def _noun_verb_balance_from_tags(tags: List[str]) -> float:
        nouns = 0
        verbs = 0
        for flag in tags:
            if flag.startswith("n"):
                nouns += 1
            elif flag.startswith("v"):
                verbs += 1
        total = nouns + verbs
        if total == 0:
            return 0.5
        return nouns / total
