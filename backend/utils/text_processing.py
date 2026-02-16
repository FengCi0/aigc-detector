import logging
import os
import re
from logging.handlers import RotatingFileHandler
from typing import List, Optional


def _parse_log_level(raw_level: str) -> int:
    if not raw_level:
        return logging.INFO
    return getattr(logging, raw_level.upper(), logging.INFO)


def setup_logging(log_level: Optional[int] = None) -> logging.Logger:
    """设置日志，避免重复挂载handler。"""
    resolved_level = log_level if log_level is not None else _parse_log_level(os.getenv("AIGC_LOG_LEVEL", "INFO"))
    logger = logging.getLogger()
    logger.setLevel(resolved_level)

    if getattr(logger, "_aigc_configured", False):
        return logger

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "aigc_detector.log")

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger._aigc_configured = True  # type: ignore[attr-defined]
    return logger


def preprocess_text(text: str, preserve_newlines: bool = False) -> str:
    """预处理文本。

    preserve_newlines=True 时保留段落边界，供长文本分段使用。
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u00a0", " ")

    allowed = r"[^\w\s\u4e00-\u9fff，。；：！？、“”‘’《》【】（）\(\)\[\]\.,:;!?\"'\-—…]"
    text = re.sub(allowed, "", text)

    if preserve_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
    else:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def _split_paragraph_to_sentences(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;])", paragraph)
    sentences = [s.strip() for s in parts if s and s.strip()]
    return sentences if sentences else [paragraph]


def split_text_into_chunks(text: str, max_chunk_size: int = 1000, min_chunk_size: int = 120) -> List[str]:
    """将长文本分割为多个块，优先按段落和句子边界切分。"""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    def append_piece(piece: str) -> None:
        nonlocal current
        if not piece:
            return
        if not current:
            current = piece
            return
        candidate = f"{current}\n{piece}"
        if len(candidate) <= max_chunk_size:
            current = candidate
        else:
            flush_current()
            current = piece

    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_size:
            append_piece(paragraph)
            continue

        sentences = _split_paragraph_to_sentences(paragraph)
        sentence_buffer = ""
        for sentence in sentences:
            if len(sentence) > max_chunk_size:
                if sentence_buffer:
                    append_piece(sentence_buffer)
                    sentence_buffer = ""
                for i in range(0, len(sentence), max_chunk_size):
                    append_piece(sentence[i : i + max_chunk_size])
                continue

            if not sentence_buffer:
                sentence_buffer = sentence
            elif len(sentence_buffer) + len(sentence) <= max_chunk_size:
                sentence_buffer += sentence
            else:
                append_piece(sentence_buffer)
                sentence_buffer = sentence

        if sentence_buffer:
            append_piece(sentence_buffer)

    flush_current()

    if len(chunks) <= 1:
        return chunks

    merged: List[str] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        if len(chunk) < min_chunk_size and len(merged[-1]) + len(chunk) + 1 <= max_chunk_size:
            merged[-1] = f"{merged[-1]}\n{chunk}"
        else:
            merged.append(chunk)
    return merged
