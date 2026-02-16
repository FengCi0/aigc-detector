import os
from pathlib import Path
from typing import List, Union

from flask import Flask, Response, jsonify, send_from_directory
from flask_cors import CORS

try:
    from backend.api.routes import api_bp, detector
    from backend.utils.text_processing import setup_logging
except ImportError:  # 兼容从backend目录直接运行
    from api.routes import api_bp, detector
    from utils.text_processing import setup_logging


def _parse_origins(raw: str) -> Union[str, List[str]]:
    if not raw or raw.strip() == "*":
        return "*"
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _frontend_build_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "frontend" / "build"


def _resolve_max_content_length() -> int:
    raw = os.getenv("AIGC_MAX_CONTENT_LENGTH", str(256 * 1024))
    try:
        return max(int(raw), 32 * 1024)
    except ValueError:
        return 256 * 1024


def create_app() -> Flask:
    # 关闭 Flask 默认 /static 路由，避免拦截前端 build 静态资源请求
    app = Flask(__name__, static_folder=None)
    setup_logging()
    app.config["MAX_CONTENT_LENGTH"] = _resolve_max_content_length()

    origins = _parse_origins(os.getenv("AIGC_ALLOWED_ORIGINS", "*"))
    CORS(app, resources={r"/api/*": {"origins": origins}})

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        return (
            jsonify(
                {
                    "error": {
                        "code": "payload_too_large",
                        "message": "请求体过大，请缩短文本后重试",
                    }
                }
            ),
            413,
        )

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify(
            {
                "status": "ok",
                "service": "aigc-detector",
                "model_mode": detector.model_mode,
                "model_loaded": detector.model_loaded,
                "min_text_length": detector.min_text_length,
                "transformer_enabled": bool(getattr(detector, "transformer_enabled", False)),
                "transformer_model_id": getattr(detector, "transformer_model_id", None),
                "transformer_apply_to": getattr(detector, "transformer_apply_to", "all"),
                "non_zh_score_threshold": getattr(detector, "non_zh_score_threshold", None),
            }
        )

    @app.route("/", defaults={"path": ""}, methods=["GET"])
    @app.route("/<path:path>", methods=["GET"])
    def serve_frontend(path: str):
        build_dir = _frontend_build_dir()
        if build_dir.exists():
            requested = build_dir / path
            if path and requested.is_file():
                return send_from_directory(str(build_dir), path)

            index_html = build_dir / "index.html"
            if index_html.exists():
                return send_from_directory(str(build_dir), "index.html")

        return Response(
            (
                "AIGC Detector backend is running.\n"
                "Available endpoints:\n"
                "GET /health\n"
                "POST /api/detect\n"
                "Tip: run `cd frontend && npm run build` to enable browser UI on this port.\n"
            ),
            mimetype="text/plain; charset=utf-8",
        )

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = _env_flag("FLASK_DEBUG", "false")
    app.run(host="0.0.0.0", port=port, debug=debug)
