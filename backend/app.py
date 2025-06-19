from flask import Flask, jsonify
from flask_cors import CORS
from api.routes import api_bp
import os
from utils.text_processing import setup_logging

# 设置应用
app = Flask(__name__)
CORS(app)

# 设置日志
logger = setup_logging()

# 注册蓝图
app.register_blueprint(api_bp, url_prefix='/api')

# 简单的健康检查路由
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "AIGC检测器服务运行中"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 