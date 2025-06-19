import re
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO):
    """设置日志

    Args:
        log_level: 日志级别

    Returns:
        logger: 日志器实例
    """
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'aigc_detector.log')
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 设置文件处理器
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # 设置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 获取根日志器并配置
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def preprocess_text(text):
    """预处理文本

    Args:
        text (str): 原始文本

    Returns:
        str: 预处理后的文本
    """
    if not text:
        return ""
    
    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 去除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 去除特殊字符（保留中英文字符、数字、基本标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，、；：""''！？《》【】\(\)\.,-?!]', '', text)
    
    return text

def split_text_into_chunks(text, max_chunk_size=1000):
    """将长文本分割为多个块

    Args:
        text (str): 输入文本
        max_chunk_size (int): 每块的最大字符数

    Returns:
        list: 文本块列表
    """
    # 按自然段落分割
    paragraphs = re.split(r'\n+', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # 如果当前段落加上已有内容超过最大长度，开始新块
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n" + paragraph
            else:
                current_chunk = paragraph
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks 