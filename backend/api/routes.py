import logging
import os
import time
import uuid
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Dict

from flask import Blueprint, jsonify, request

try:
    from backend.models.detector import AIGCDetector
except ImportError:  # 兼容从backend目录直接运行
    from models.detector import AIGCDetector


api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)
detector = AIGCDetector.from_env()
_rate_limit_lock = Lock()
_request_windows: Dict[str, deque] = defaultdict(deque)


def _resolve_rate_limit_per_minute() -> int:
    raw = os.getenv("AIGC_RATE_LIMIT_PER_MIN", "120")
    try:
        return max(int(raw), 10)
    except ValueError:
        return 120


def _is_rate_limited(client_key: str) -> bool:
    limit = _resolve_rate_limit_per_minute()
    now = time.time()
    window_start = now - 60.0
    with _rate_limit_lock:
        q = _request_windows[client_key]
        while q and q[0] < window_start:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
    return False


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _resolve_max_text_length() -> int:
    raw = os.getenv("AIGC_MAX_TEXT_LENGTH", "20000")
    try:
        return max(int(raw), 500)
    except ValueError:
        return 20000


def _error_response(message: str, status_code: int, code: str) -> tuple:
    payload = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    return jsonify(payload), status_code


@api_bp.route("/detect", methods=["POST"])
def detect_aigc():
    start_time = time.time()

    client_key = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip() or "unknown"
    if _is_rate_limited(client_key):
        return _error_response("请求过于频繁，请稍后重试", 429, "rate_limited")

    if not request.is_json:
        return _error_response("请求必须是JSON格式", 415, "invalid_content_type")

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    text = payload.get("text")
    include_details = _parse_bool(payload.get("include_details", False), default=False)

    if not isinstance(text, str) or not text.strip():
        return _error_response("text 字段不能为空字符串", 400, "invalid_text")
    max_text_length = _resolve_max_text_length()
    if len(text) > max_text_length:
        return _error_response(
            f"文本长度超过限制，最多允许 {max_text_length} 个字符",
            400,
            "validation_error",
        )

    try:
        result = detector.analyze(text, include_details=include_details)
        processing_time = round(time.time() - start_time, 3)

        response = {
            "aigc_score": result["aigc_score"],
            "confidence": result["confidence"],
            "label": result["label"],
            "score_threshold": result["score_threshold"],
            "features": result["features"],
            "processing_time": processing_time,
            "model_mode": result["model_mode"],
            "warnings": result.get("warnings", []),
        }
        if include_details and "details" in result:
            response["details"] = result["details"]

        logger.info(
            "检测完成 score=%.2f confidence=%.2f mode=%s processing_time=%.3fs",
            float(result["aigc_score"]),
            float(result["confidence"]),
            result["model_mode"],
            processing_time,
        )
        return jsonify(response)
    except ValueError as exc:
        return _error_response(str(exc), 400, "validation_error")
    except Exception as exc:
        request_id = str(uuid.uuid4())
        logger.exception("检测请求失败 request_id=%s error=%s", request_id, exc)
        return _error_response(f"服务器内部错误，请稍后重试（request_id={request_id}）", 500, "internal_error")
