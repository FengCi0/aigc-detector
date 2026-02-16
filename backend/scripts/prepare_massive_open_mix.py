#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
构建“大规模公开语料混合集”用于 AIGC 检测训练。

输出目录结构:
data/dataset/
  ├── ai/
  └── human/

支持来源（可组合）:
- hc3_zh: Hello-SimpleAI/HC3-Chinese（JSONL直连）
- hc3_en: Hello-SimpleAI/HC3（JSONL直连）
- semeval24_mono: d0rj/SemEval2024-task8
- daigt: Yunij/kaggle-comp-daigt
- mage: yaful/MAGE
- raid: liamdugan/raid
- wildchat_zh: SUSTech/wildchat_zh
"""

import argparse
import json
import os
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.utils.text_processing import preprocess_text


HC3_URLS: Dict[str, str] = {
    "hc3_zh": "https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/resolve/main/all.jsonl?download=true",
    "hc3_en": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl?download=true",
}


def _iter_jsonl(url: str) -> Iterable[dict]:
    with urllib.request.urlopen(url, timeout=90) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _clean(text: str, min_chars: int, max_chars: int) -> Optional[str]:
    cleaned = preprocess_text(text, preserve_newlines=True)
    if len(cleaned) < min_chars:
        return None
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def _write_texts(texts: List[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, text in enumerate(texts, start=1):
        (out_dir / f"{prefix}_{idx:07d}.txt").write_text(text, encoding="utf-8")


def _collect_hc3(
    source: str,
    per_class_cap: int,
    min_chars: int,
    max_chars: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    ai_texts: List[str] = []
    human_texts: List[str] = []
    seen_ai = set()
    seen_human = set()
    for row in _iter_jsonl(HC3_URLS[source]):
        human_answers = row.get("human_answers") or []
        ai_answers = row.get("chatgpt_answers") or []

        if len(human_texts) < per_class_cap:
            for ans in human_answers:
                cleaned = _clean(str(ans), min_chars, max_chars)
                if not cleaned or cleaned in seen_human:
                    continue
                seen_human.add(cleaned)
                human_texts.append(cleaned)
                break

        if len(ai_texts) < per_class_cap:
            for ans in ai_answers:
                cleaned = _clean(str(ans), min_chars, max_chars)
                if not cleaned or cleaned in seen_ai:
                    continue
                seen_ai.add(cleaned)
                ai_texts.append(cleaned)
                break

        if len(ai_texts) >= per_class_cap and len(human_texts) >= per_class_cap:
            break
    return ai_texts, human_texts


def _collect_semeval(
    per_class_cap: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    max_rows: int,
) -> Tuple[List[str], List[str]]:
    ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), max_rows)))

    ai_texts: List[str] = []
    human_texts: List[str] = []
    seen_ai = set()
    seen_human = set()
    for row in ds:
        label = int(row.get("label", -1))
        text = _clean(str(row.get("text", "")), min_chars, max_chars)
        if not text:
            continue
        if label == 1 and len(ai_texts) < per_class_cap and text not in seen_ai:
            seen_ai.add(text)
            ai_texts.append(text)
        elif label == 0 and len(human_texts) < per_class_cap and text not in seen_human:
            seen_human.add(text)
            human_texts.append(text)
        if len(ai_texts) >= per_class_cap and len(human_texts) >= per_class_cap:
            break
    return ai_texts, human_texts


def _collect_daigt(
    per_class_cap: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    max_rows: int,
) -> Tuple[List[str], List[str]]:
    ds = load_dataset("Yunij/kaggle-comp-daigt", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), max_rows)))

    ai_texts: List[str] = []
    human_texts: List[str] = []
    seen_ai = set()
    seen_human = set()
    for row in ds:
        label = int(row.get("label", -1))
        text = _clean(str(row.get("text", "")), min_chars, max_chars)
        if not text:
            continue
        if label == 1 and len(ai_texts) < per_class_cap and text not in seen_ai:
            seen_ai.add(text)
            ai_texts.append(text)
        elif label == 0 and len(human_texts) < per_class_cap and text not in seen_human:
            seen_human.add(text)
            human_texts.append(text)
        if len(ai_texts) >= per_class_cap and len(human_texts) >= per_class_cap:
            break
    return ai_texts, human_texts


def _collect_mage(
    per_class_cap: int,
    min_chars: int,
    max_chars: int,
    seed: int,
    max_rows: int,
) -> Tuple[List[str], List[str]]:
    ds = load_dataset("yaful/MAGE", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), max_rows)))

    # MAGE 标签约定：label=0 -> machine, label=1 -> human
    ai_texts: List[str] = []
    human_texts: List[str] = []
    seen_ai = set()
    seen_human = set()
    for row in ds:
        label = int(row.get("label", -1))
        text = _clean(str(row.get("text", "")), min_chars, max_chars)
        if not text:
            continue
        if label == 0 and len(ai_texts) < per_class_cap and text not in seen_ai:
            seen_ai.add(text)
            ai_texts.append(text)
        elif label == 1 and len(human_texts) < per_class_cap and text not in seen_human:
            seen_human.add(text)
            human_texts.append(text)
        if len(ai_texts) >= per_class_cap and len(human_texts) >= per_class_cap:
            break
    return ai_texts, human_texts


def _collect_raid(
    per_class_cap: int,
    min_chars: int,
    max_chars: int,
    max_rows: int,
) -> Tuple[List[str], List[str]]:
    # RAID 很大，采用 streaming 防止全量落盘
    ds = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)

    ai_texts: List[str] = []
    human_texts: List[str] = []
    seen_ai = set()
    seen_human = set()
    row_count = 0
    for row in ds:
        row_count += 1
        if row_count > max_rows:
            break
        model_name = str(row.get("model", "")).strip().lower()
        text = _clean(str(row.get("generation", "")), min_chars, max_chars)
        if not text:
            continue
        if model_name == "human":
            if len(human_texts) < per_class_cap and text not in seen_human:
                seen_human.add(text)
                human_texts.append(text)
        else:
            if len(ai_texts) < per_class_cap and text not in seen_ai:
                seen_ai.add(text)
                ai_texts.append(text)
        if len(ai_texts) >= per_class_cap and len(human_texts) >= per_class_cap:
            break
    return ai_texts, human_texts


def _collect_wildchat_zh(
    per_class_cap: int,
    min_chars: int,
    max_chars: int,
    max_rows: int,
) -> Tuple[List[str], List[str]]:
    ds = load_dataset("SUSTech/wildchat_zh", split="train", streaming=True)

    ai_texts: List[str] = []
    human_texts: List[str] = []
    seen_ai = set()
    seen_human = set()
    row_count = 0
    for row in ds:
        row_count += 1
        if row_count > max_rows:
            break
        conv = row.get("conversations") or []
        if not isinstance(conv, list):
            continue
        for turn in conv:
            role = str(turn.get("role", "")).strip().lower()
            lang = str(turn.get("language", "")).strip().lower()
            if "chinese" not in lang:
                continue
            if bool(turn.get("redacted", False)):
                continue
            cleaned = _clean(str(turn.get("content", "")), min_chars, max_chars)
            if not cleaned:
                continue
            if role == "assistant":
                if len(ai_texts) < per_class_cap and cleaned not in seen_ai:
                    seen_ai.add(cleaned)
                    ai_texts.append(cleaned)
            elif role == "user":
                if len(human_texts) < per_class_cap and cleaned not in seen_human:
                    seen_human.add(cleaned)
                    human_texts.append(cleaned)
            if len(ai_texts) >= per_class_cap and len(human_texts) >= per_class_cap:
                break
        if len(ai_texts) >= per_class_cap and len(human_texts) >= per_class_cap:
            break
    return ai_texts, human_texts


COLLECTORS = {
    "hc3_zh": _collect_hc3,
    "hc3_en": _collect_hc3,
    "semeval24_mono": _collect_semeval,
    "daigt": _collect_daigt,
    "mage": _collect_mage,
    "raid": _collect_raid,
    "wildchat_zh": _collect_wildchat_zh,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="构建多源公开训练集（大规模）")
    parser.add_argument(
        "--sources",
        default="hc3_zh,hc3_en,semeval24_mono,daigt,mage,raid,wildchat_zh",
        help="逗号分隔来源",
    )
    parser.add_argument("--per-source-cap", type=int, default=12000, help="每个来源每类最多抽取样本数")
    parser.add_argument("--max-rows-per-source", type=int, default=500000, help="每个来源最多扫描行数")
    parser.add_argument("--target-per-class", type=int, default=60000, help="最终每类样本上限")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="data/dataset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-on-source-error",
        action="store_true",
        help="默认遇到单个来源失败会跳过并继续；开启后改为立即失败",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise RuntimeError(f"输出目录已存在: {output_root}，如需覆盖请加 --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_names = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in source_names:
        if s not in COLLECTORS:
            raise ValueError(f"未知来源: {s}，可选={list(COLLECTORS.keys())}")

    all_ai: List[str] = []
    all_human: List[str] = []
    seen_ai = set()
    seen_human = set()
    stats: Dict[str, Dict[str, int]] = {}

    for source in source_names:
        collector = COLLECTORS[source]
        print(f"[collect] source={source}")
        try:
            if source in {"hc3_zh", "hc3_en"}:
                ai_texts, human_texts = collector(  # type: ignore[misc]
                    source=source,
                    per_class_cap=int(args.per_source_cap),
                    min_chars=int(args.min_chars),
                    max_chars=int(args.max_chars),
                    seed=int(args.seed),
                )
            elif source in {"semeval24_mono", "daigt", "mage"}:
                ai_texts, human_texts = collector(  # type: ignore[misc]
                    per_class_cap=int(args.per_source_cap),
                    min_chars=int(args.min_chars),
                    max_chars=int(args.max_chars),
                    seed=int(args.seed),
                    max_rows=int(args.max_rows_per_source),
                )
            else:
                ai_texts, human_texts = collector(  # type: ignore[misc]
                    per_class_cap=int(args.per_source_cap),
                    min_chars=int(args.min_chars),
                    max_chars=int(args.max_chars),
                    max_rows=int(args.max_rows_per_source),
                )
        except Exception as exc:
            stats[source] = {
                "raw_ai": 0,
                "raw_human": 0,
                "added_ai": 0,
                "added_human": 0,
                "error": str(exc),
            }
            print(f"[collect] {source}: ERROR={exc}")
            if args.fail_on_source_error:
                raise
            continue

        added_ai = 0
        for t in ai_texts:
            if t in seen_ai:
                continue
            seen_ai.add(t)
            all_ai.append(t)
            added_ai += 1
        added_human = 0
        for t in human_texts:
            if t in seen_human:
                continue
            seen_human.add(t)
            all_human.append(t)
            added_human += 1

        stats[source] = {
            "raw_ai": len(ai_texts),
            "raw_human": len(human_texts),
            "added_ai": added_ai,
            "added_human": added_human,
        }
        print(f"[collect] {source}: raw_ai={len(ai_texts)} raw_human={len(human_texts)}")

    class_size = min(len(all_ai), len(all_human), int(args.target_per_class))
    if class_size < 1000:
        raise RuntimeError(f"有效样本不足: ai={len(all_ai)} human={len(all_human)}")

    random.shuffle(all_ai)
    random.shuffle(all_human)
    all_ai = all_ai[:class_size]
    all_human = all_human[:class_size]

    _write_texts(all_ai, output_root / "ai", "ai")
    _write_texts(all_human, output_root / "human", "human")

    summary = {
        "sources": source_names,
        "per_source_cap": int(args.per_source_cap),
        "max_rows_per_source": int(args.max_rows_per_source),
        "samples_per_class": class_size,
        "min_chars": int(args.min_chars),
        "max_chars": int(args.max_chars),
        "source_stats": stats,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"输出目录: {output_root.resolve()}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception as exc:
        code = 1
        print(f"[fatal] {exc}", file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
