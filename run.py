#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目统一入口：
1) 启动服务：python run.py
2) 训练轻量模型：python run.py train --ai-data <dir> --human-data <dir>
3) 训练 Transformer：python run.py train-transformer --ai-data <dir> --human-data <dir>
4) 一键产品训练：python run.py train-full --ai-data <dir> --human-data <dir>
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "data" / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "aigc_detector_model.joblib"
DEFAULT_TRANSFORMER_DIR = MODELS_DIR / "transformer_detector"
RUNTIME_CONFIG_PATH = MODELS_DIR / "runtime_config.json"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "dataset"
DEFAULT_AI_DIR = DEFAULT_DATASET_DIR / "ai"
DEFAULT_HUMAN_DIR = DEFAULT_DATASET_DIR / "human"


def _run_command(cmd: list[str]) -> int:
    print("执行命令:", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(completed.returncode)


def _load_runtime_env_defaults() -> dict[str, str]:
    if not RUNTIME_CONFIG_PATH.exists():
        return {}
    try:
        payload = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    env = payload.get("env", {})
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items() if v is not None}


def _apply_serve_runtime_defaults() -> None:
    runtime_defaults = _load_runtime_env_defaults()
    for key, value in runtime_defaults.items():
        os.environ.setdefault(key, value)

    # 若已有本地 Transformer 模型但还没有 runtime_config，则使用“大模型主导”默认模式。
    if not runtime_defaults and (DEFAULT_TRANSFORMER_DIR / "config.json").exists():
        os.environ.setdefault("AIGC_ENABLE_TRANSFORMER", "1")
        os.environ.setdefault("AIGC_TRANSFORMER_MODEL_ID", str(DEFAULT_TRANSFORMER_DIR))
        os.environ.setdefault("AIGC_TRANSFORMER_APPLY_TO", "all")
        os.environ.setdefault("AIGC_TRANSFORMER_WEIGHT", "1.0")
        os.environ.setdefault("AIGC_TRANSFORMER_CJK_THRESHOLD", "0.2")


def _write_runtime_config_from_tune_report(report_path: Path, transformer_model_id: str) -> Optional[Path]:
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    recommended_env = report.get("recommended_env", {})
    if not isinstance(recommended_env, dict):
        recommended_env = {}
    recommended_env["AIGC_TRANSFORMER_MODEL_ID"] = transformer_model_id
    recommended_env.setdefault("AIGC_TRANSFORMER_APPLY_TO", "all")
    weight_raw = str(recommended_env.get("AIGC_TRANSFORMER_WEIGHT", "0.0"))
    try:
        weight = float(weight_raw)
    except ValueError:
        weight = 0.0
    if weight <= 0.01:
        recommended_env["AIGC_ENABLE_TRANSFORMER"] = "0"
    else:
        recommended_env["AIGC_ENABLE_TRANSFORMER"] = "1"

    runtime_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "train-full",
        "env": recommended_env,
        "notes": "服务启动时将自动使用该配置。",
    }
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG_PATH.write_text(json.dumps(runtime_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return RUNTIME_CONFIG_PATH


def _resolve_data_dirs(ai_data: str, human_data: str) -> tuple[Path, Path]:
    ai_dir = Path(ai_data) if ai_data else DEFAULT_AI_DIR
    human_dir = Path(human_data) if human_data else DEFAULT_HUMAN_DIR
    return ai_dir, human_dir


def _validate_data_dirs(ai_dir: Path, human_dir: Path) -> bool:
    if not ai_dir.exists() or not human_dir.exists():
        print("训练数据目录不存在，请准备 data/dataset/ai 和 data/dataset/human，或显式指定 --ai-data/--human-data")
        print(f"当前 AI 目录: {ai_dir}")
        print(f"当前 Human 目录: {human_dir}")
        return False
    ai_count = len(list(ai_dir.rglob("*.txt")))
    human_count = len(list(human_dir.rglob("*.txt")))
    if ai_count <= 0 or human_count <= 0:
        print("训练数据为空，请确保 ai/human 目录中均包含 .txt 文件。")
        print(f"当前 AI 样本数: {ai_count}")
        print(f"当前 Human 样本数: {human_count}")
        return False
    return True


def _light_train_cmd(
    ai_dir: Path,
    human_dir: Path,
    passthrough: list[str],
    max_files: Optional[int] = None,
    min_length: Optional[int] = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "backend" / "train_model.py"),
        "--ai-data",
        str(ai_dir),
        "--human-data",
        str(human_dir),
        "--output",
        str(DEFAULT_MODEL_PATH),
    ]
    if max_files is not None and max_files > 0:
        cmd.extend(["--max-files", str(max_files)])
    if min_length is not None and min_length > 0:
        cmd.extend(["--min-length", str(min_length)])
    cmd.extend(passthrough)
    return cmd


def _transformer_train_cmd(ai_dir: Path, human_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "backend" / "train_transformer.py"),
        "--ai-data",
        str(ai_dir),
        "--human-data",
        str(human_dir),
        "--output-dir",
        str(DEFAULT_TRANSFORMER_DIR),
        "--base-model",
        args.base_model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--min-length",
        str(args.min_length),
        "--max-files",
        str(args.max_files) if args.max_files is not None else "0",
        "--seed",
        str(args.seed),
    ]


def serve(args: argparse.Namespace) -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AIGC_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    _apply_serve_runtime_defaults()

    port = getattr(args, "port", None)
    debug = bool(getattr(args, "debug", False))
    if port is not None:
        os.environ["PORT"] = str(port)
    if debug:
        os.environ["FLASK_DEBUG"] = "1"

    from backend.app import create_app

    app = create_app()
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    print(
        "启动AIGC检测服务..."
        f" port={port}"
        f" debug={debug}"
        f" model={os.getenv('AIGC_MODEL_PATH')}"
        f" transformer_enabled={os.getenv('AIGC_ENABLE_TRANSFORMER', '0')}",
        flush=True,
    )
    app.run(host="0.0.0.0", port=port, debug=debug)
    return 0


def train(args: argparse.Namespace, passthrough: list[str]) -> int:
    ai_dir, human_dir = _resolve_data_dirs(args.ai_data, args.human_data)
    if not _validate_data_dirs(ai_dir, human_dir):
        return 2
    print(f"开始训练轻量模型，固定模型路径: {DEFAULT_MODEL_PATH}", flush=True)
    return _run_command(
        _light_train_cmd(
            ai_dir,
            human_dir,
            passthrough,
            max_files=args.max_files,
            min_length=args.min_length,
        )
    )


def train_transformer(args: argparse.Namespace) -> int:
    ai_dir, human_dir = _resolve_data_dirs(args.ai_data, args.human_data)
    if not _validate_data_dirs(ai_dir, human_dir):
        return 2
    print(f"开始训练 Transformer 模型，输出目录: {DEFAULT_TRANSFORMER_DIR}", flush=True)
    return _run_command(_transformer_train_cmd(ai_dir, human_dir, args))


def train_full(args: argparse.Namespace, passthrough: list[str]) -> int:
    ai_dir, human_dir = _resolve_data_dirs(args.ai_data, args.human_data)
    if not _validate_data_dirs(ai_dir, human_dir):
        return 2

    print("第一步：训练轻量主模型", flush=True)
    rc = _run_command(
        _light_train_cmd(
            ai_dir,
            human_dir,
            passthrough,
            max_files=args.max_files,
            min_length=args.min_length,
        )
    )
    if rc != 0:
        return rc

    print("第二步：训练 Transformer 分支", flush=True)
    rc = _run_command(_transformer_train_cmd(ai_dir, human_dir, args))
    if rc != 0:
        return rc

    if args.skip_tune:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "train-full(skip_tune)",
            "env": {
                "AIGC_ENABLE_TRANSFORMER": "1",
                "AIGC_TRANSFORMER_MODEL_ID": str(DEFAULT_TRANSFORMER_DIR),
                "AIGC_TRANSFORMER_APPLY_TO": "all",
                "AIGC_TRANSFORMER_WEIGHT": "1.0",
            },
        }
        RUNTIME_CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入默认运行配置: {RUNTIME_CONFIG_PATH}", flush=True)
        return 0

    print("第三步：搜索融合权重并生成上线配置", flush=True)
    tune_report_path = MODELS_DIR / "tune_transformer_blend_report.json"
    tune_cmd = [
        sys.executable,
        str(REPO_ROOT / "backend" / "scripts" / "tune_transformer_blend.py"),
        "--model-path",
        str(DEFAULT_MODEL_PATH),
        "--ai-dir",
        str(ai_dir),
        "--human-dir",
        str(human_dir),
        "--transformer-model-id",
        str(DEFAULT_TRANSFORMER_DIR),
        "--max-samples",
        str(args.tune_max_samples),
        "--output",
        str(tune_report_path),
    ]
    rc = _run_command(tune_cmd)
    if rc != 0:
        return rc

    runtime_path = _write_runtime_config_from_tune_report(
        report_path=tune_report_path,
        transformer_model_id=str(DEFAULT_TRANSFORMER_DIR),
    )
    if runtime_path is None:
        print("警告：融合报告解析失败，未生成 runtime_config.json", flush=True)
        return 1
    print(f"训练完成，已生成运行配置: {runtime_path}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIGC Detector 统一入口")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="启动服务")
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--debug", action="store_true")

    train_parser = sub.add_parser("train", help="训练并迭代轻量主模型")
    train_parser.add_argument("--ai-data", default=str(DEFAULT_AI_DIR))
    train_parser.add_argument("--human-data", default=str(DEFAULT_HUMAN_DIR))
    train_parser.add_argument("--max-files", type=int, default=None)
    train_parser.add_argument("--min-length", type=int, default=50)

    transformer_parser = sub.add_parser("train-transformer", help="训练 Transformer 分支模型")
    transformer_parser.add_argument("--ai-data", default=str(DEFAULT_AI_DIR))
    transformer_parser.add_argument("--human-data", default=str(DEFAULT_HUMAN_DIR))
    transformer_parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    transformer_parser.add_argument("--epochs", type=float, default=2.0)
    transformer_parser.add_argument("--batch-size", type=int, default=8)
    transformer_parser.add_argument("--learning-rate", type=float, default=2e-5)
    transformer_parser.add_argument("--max-length", type=int, default=384)
    transformer_parser.add_argument("--min-length", type=int, default=50)
    transformer_parser.add_argument("--max-files", type=int, default=None)
    transformer_parser.add_argument("--seed", type=int, default=42)

    full_parser = sub.add_parser("train-full", help="一键训练产品模型（轻量 + Transformer + 融合配置）")
    full_parser.add_argument("--ai-data", default=str(DEFAULT_AI_DIR))
    full_parser.add_argument("--human-data", default=str(DEFAULT_HUMAN_DIR))
    full_parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    full_parser.add_argument("--epochs", type=float, default=2.0)
    full_parser.add_argument("--batch-size", type=int, default=8)
    full_parser.add_argument("--learning-rate", type=float, default=2e-5)
    full_parser.add_argument("--max-length", type=int, default=384)
    full_parser.add_argument("--min-length", type=int, default=50)
    full_parser.add_argument("--max-files", type=int, default=None)
    full_parser.add_argument("--seed", type=int, default=42)
    full_parser.add_argument("--tune-max-samples", type=int, default=1200)
    full_parser.add_argument("--skip-tune", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args()
    if args.command in (None, "serve"):
        return serve(args)
    if args.command == "train":
        return train(args, passthrough)
    if args.command == "train-transformer":
        return train_transformer(args)
    if args.command == "train-full":
        return train_full(args, passthrough)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
