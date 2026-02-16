import os

try:
    from backend.app import create_app
except ImportError:  # 兼容在backend目录执行 python run.py
    from app import create_app


app = create_app()


if __name__ == "__main__":
    print("提示：推荐在项目根目录使用 `python run.py` 统一启动。")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    print(f"启动AIGC检测服务... port={port} debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
