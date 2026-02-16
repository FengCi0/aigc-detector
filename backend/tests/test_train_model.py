import numpy as np

from backend.train_model import augment_short_texts


def test_augment_short_texts_returns_expected_shapes():
    texts = [
        "这是第一句。这是第二句。这是第三句，用于测试增强逻辑。",
        "另一个样本，包含多句。为了测试句子删除增强，需要足够长度。再来一句。",
    ]
    labels = np.array([1, 0], dtype=int)

    aug_texts, aug_labels, aug_count = augment_short_texts(
        texts=texts,
        labels=labels,
        ratio=1.0,
        min_len=10,
        max_len=80,
        sentence_drop_prob=0.3,
        seed=42,
    )

    assert len(aug_texts) == len(aug_labels)
    assert aug_count > 0
    assert len(aug_texts) >= len(texts)


def test_augment_short_texts_ratio_zero_no_changes():
    texts = ["这是一个样本，用于测试。", "这是第二个样本，用于测试。"]
    labels = np.array([1, 0], dtype=int)
    aug_texts, aug_labels, aug_count = augment_short_texts(texts, labels, ratio=0.0, seed=42)

    assert aug_count == 0
    assert aug_texts == texts
    assert np.array_equal(aug_labels, labels)
