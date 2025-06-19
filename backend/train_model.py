#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练AIGC检测模型脚本
用法: python train_model.py --ai-data path/to/ai/texts --human-data path/to/human/texts
"""

import os
import argparse
import glob
import random
import logging
from models.detector import AIGCDetector
from utils.text_processing import setup_logging, preprocess_text

# 设置日志
logger = setup_logging()

def load_text_files(directory, file_pattern="*.txt", max_files=None, min_length=50):
    """加载指定目录下的文本文件
    
    Args:
        directory (str): 文本文件目录
        file_pattern (str): 文件匹配模式
        max_files (int): 最大加载文件数
        min_length (int): 最小文本长度
        
    Returns:
        list: 文本内容列表
    """
    if not os.path.exists(directory):
        logger.error(f"目录不存在: {directory}")
        return []
        
    pattern = os.path.join(directory, file_pattern)
    files = glob.glob(pattern)
    
    if max_files and len(files) > max_files:
        files = random.sample(files, max_files)
        
    texts = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 预处理文本
                content = preprocess_text(content)
                
                if len(content) >= min_length:
                    texts.append(content)
                else:
                    logger.warning(f"文件 {file_path} 内容过短，已跳过")
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {str(e)}")
            
    logger.info(f"从 {directory} 加载了 {len(texts)} 个文本")
    return texts

def main():
    parser = argparse.ArgumentParser(description='训练AIGC检测模型')
    parser.add_argument('--ai-data', required=True, help='AI生成文本数据目录')
    parser.add_argument('--human-data', required=True, help='人工撰写文本数据目录')
    parser.add_argument('--output', default='../data/models/aigc_detector_model.joblib', help='模型输出路径')
    parser.add_argument('--max-files', type=int, default=None, help='每类最大文件数')
    parser.add_argument('--min-length', type=int, default=50, help='最小文本长度')
    parser.add_argument('--file-pattern', default='*.txt', help='文件匹配模式')
    parser.add_argument('--use-advanced', action='store_true', help='是否使用高级特征')
    
    args = parser.parse_args()
    
    # 加载数据
    logger.info("开始加载训练数据...")
    ai_texts = load_text_files(
        args.ai_data, 
        file_pattern=args.file_pattern,
        max_files=args.max_files,
        min_length=args.min_length
    )
    
    human_texts = load_text_files(
        args.human_data, 
        file_pattern=args.file_pattern,
        max_files=args.max_files,
        min_length=args.min_length
    )
    
    if not ai_texts or not human_texts:
        logger.error("数据加载失败，请检查路径和文件")
        return
        
    logger.info(f"加载了 {len(ai_texts)} 个AI文本和 {len(human_texts)} 个人工文本")
    
    # 准备训练数据
    texts = ai_texts + human_texts
    labels = [1] * len(ai_texts) + [0] * len(human_texts)
    
    # 随机打乱数据
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    # 初始化检测器并训练
    logger.info("开始训练模型...")
    detector = AIGCDetector(use_transformers=args.use_advanced)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    accuracy = detector.train(
        texts=texts,
        labels=labels,
        model_save_path=args.output,
        use_advanced_features=args.use_advanced
    )
    
    logger.info(f"模型训练完成，准确率: {accuracy:.4f}")
    logger.info(f"模型已保存至: {args.output}")

if __name__ == "__main__":
    main() 