from flask import Blueprint, request, jsonify
import time
from models.detector import AIGCDetector
import logging

# 创建蓝图
api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# 实例化检测器
detector = AIGCDetector()

@api_bp.route('/detect', methods=['POST'])
def detect_aigc():
    """检测文本的AIGC率"""
    start_time = time.time()
    
    if not request.is_json:
        return jsonify({"error": "请求必须是JSON格式"}), 400
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "未提供文本内容"}), 400
    
    try:
        # 获取文本AIGC率
        result = detector.analyze(text)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        response = {
            "aigc_score": result["aigc_score"],
            "confidence": result["confidence"],
            "features": result["features"],
            "processing_time": round(processing_time, 3)
        }
        
        logger.info(f"处理请求成功，AIGC率: {result['aigc_score']:.2f}，置信度: {result['confidence']:.2f}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": f"处理请求时发生错误: {str(e)}"}), 500 