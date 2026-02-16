from backend.utils.text_processing import preprocess_text, split_text_into_chunks


def test_preprocess_preserves_newlines_when_requested():
    raw = "第一段。\n\n第二段 https://example.com <b>HTML</b>"
    cleaned = preprocess_text(raw, preserve_newlines=True)
    assert "\n" in cleaned
    assert "http" not in cleaned
    assert "<b>" not in cleaned


def test_split_text_into_chunks_hard_splits_very_long_sentence():
    text = "这是一段很长的文本" * 500
    chunks = split_text_into_chunks(text, max_chunk_size=120, min_chunk_size=30)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
