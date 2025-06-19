import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
import os
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from models.features import FeatureExtractor

logger = logging.getLogger(__name__)

class AIGCDetector:
    """AIGC文本检测器类 - 增强版"""
    
    def __init__(self, model_path=None, use_transformers=True):
        """初始化检测器

        Args:
            model_path (str, optional): 预训练模型路径. 如果为None则使用基础特征分析.
            use_transformers (bool): 是否使用transformer模型进行检测
        """
        self.feature_extractor = FeatureExtractor(use_advanced_features=use_transformers)
        self.models = {}  # 多模型字典
        self.models_loaded = False
        self.use_transformers = use_transformers
        self.transformer_model = None
        self.tokenizer = None
        
        # 尝试加载预训练模型
        if model_path and os.path.exists(model_path):
            try:
                self.models = joblib.load(model_path)
                self.models_loaded = True
                logger.info(f"成功加载模型: {model_path}")
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                
        # 尝试加载transformer模型
        if use_transformers:
            self._load_transformer_model()
    
    def _load_transformer_model(self):
        """加载预训练transformer模型"""
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              '..', 'data', 'models', 'pretrained')
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            model_name = "bert-base-chinese"  # 中文BERT模型
            
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                num_labels=2  # 二分类: AI vs 人工
            )
            logger.info(f"成功加载transformer模型: {model_name}")
        except Exception as e:
            logger.error(f"加载transformer模型失败: {str(e)}")
            self.use_transformers = False
            
    def _get_transformer_prediction(self, text):
        """使用transformer模型进行预测
        
        Args:
            text (str): 输入文本
            
        Returns:
            tuple: (aigc_score, confidence)
        """
        if not self.use_transformers or not self.transformer_model or not self.tokenizer:
            return None
            
        try:
            # 对文本进行截断
            max_length = 512  # BERT最大支持长度
            
            # 准备输入
            inputs = self.tokenizer(
                text, 
                truncation=True, 
                max_length=max_length, 
                padding=True, 
                return_tensors="pt"
            )
            
            # 预测
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                
            # 获取logits并转换为概率
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
            
            # 获取AIGC概率和置信度
            aigc_score = float(probabilities[1])  # 索引1代表AIGC类别
            confidence = float(max(probabilities))
            
            return aigc_score, confidence
        except Exception as e:
            logger.error(f"Transformer预测失败: {str(e)}")
            return None
    
    def analyze(self, text, use_ensemble=True):
        """分析文本并返回AIGC率

        Args:
            text (str): 需要分析的文本
            use_ensemble (bool): 是否使用模型集成

        Returns:
            dict: 包含aigc_score, confidence和features的字典
        """
        # 文本预处理
        from utils.text_processing import preprocess_text
        text = preprocess_text(text)
        
        # 分析长文本
        if len(text) > 2000:
            logger.info(f"文本较长({len(text)}字符)，进行分段分析")
            from utils.text_processing import split_text_into_chunks
            chunks = split_text_into_chunks(text)
            
            # 对每个块进行分析
            results = []
            for chunk in chunks:
                chunk_result = self._analyze_text(chunk, use_ensemble)
                results.append(chunk_result)
            
            # 综合多个块的结果
            avg_score = sum(r["aigc_score"] for r in results) / len(results)
            avg_confidence = sum(r["confidence"] for r in results) / len(results)
            
            # 合并特征
            all_features = {}
            for r in results:
                for k, v in r["features"].items():
                    if k in all_features:
                        all_features[k] = (all_features[k] + v) / 2
                    else:
                        all_features[k] = v
            
            return {
                "aigc_score": round(avg_score, 2),
                "confidence": round(avg_confidence, 2),
                "features": all_features,
                "details": {
                    "chunk_results": results,
                    "method": "multi-chunk analysis"
                }
            }
        else:
            # 单文本分析
            return self._analyze_text(text, use_ensemble)
    
    def _analyze_text(self, text, use_ensemble=True):
        """分析单个文本块
        
        Args:
            text (str): 需要分析的文本
            use_ensemble (bool): 是否使用模型集成
            
        Returns:
            dict: 分析结果
        """
        # 提取特征
        features = self.feature_extractor.extract_features(text)
        features_dict = self.feature_extractor.get_feature_dict(features)
        
        # 存储所有预测结果
        predictions = []
        confidences = []
        methods_used = []
        
        # 1. Transformer模型预测(如果可用)
        if self.use_transformers and len(text) >= 20:
            transformer_result = self._get_transformer_prediction(text)
            if transformer_result:
                aigc_score, confidence = transformer_result
                predictions.append(aigc_score)
                confidences.append(confidence)
                methods_used.append("transformer")
        
        # 2. 如果模型已加载，使用机器学习模型预测
        if self.models_loaded and self.models:
            try:
                # 根据使用的特征数量选择合适的模型
                model_key = f"model_{len(features)}"
                if model_key in self.models:
                    model = self.models[model_key]
                    # 预测概率
                    proba = model.predict_proba([features])[0]
                    aigc_score = float(proba[1])  # AIGC的概率
                    confidence = float(max(proba))  # 预测的置信度
                    predictions.append(aigc_score)
                    confidences.append(confidence)
                    methods_used.append("ml_model")
                else:
                    logger.warning(f"没有找到匹配的模型: {model_key}")
            except Exception as e:
                logger.error(f"模型预测失败: {str(e)}")
        
        # 3. 使用启发式方法
        heuristic_score, heuristic_confidence = self._heuristic_score(features_dict)
        predictions.append(heuristic_score)
        confidences.append(heuristic_confidence)
        methods_used.append("heuristic")
        
        # 结合所有预测结果
        if use_ensemble and len(predictions) > 1:
            # 加权平均（根据置信度加权）
            weighted_sum = sum(pred * conf for pred, conf in zip(predictions, confidences))
            weights_sum = sum(confidences)
            final_score = weighted_sum / weights_sum if weights_sum > 0 else heuristic_score
            
            # 最终置信度基于预测的一致性
            score_variance = np.var(predictions)
            consistency_factor = np.exp(-5 * score_variance)  # 方差越大，一致性越低
            final_confidence = max(confidences) * consistency_factor
        else:
            # 如果只有一个预测或不使用集成，使用第一个有效预测
            final_score = predictions[0] if predictions else 0.5
            final_confidence = confidences[0] if confidences else 0.5
        
        return {
            "aigc_score": round(final_score * 100, 2),  # 转换为百分比
            "confidence": round(final_confidence * 100, 2),  # 转换为百分比
            "features": features_dict,
            "details": {
                "predictions": predictions,
                "confidences": confidences,
                "methods": methods_used
            }
        }
    
    def _heuristic_score(self, features):
        """基于特征的启发式AIGC评分，增强版

        Args:
            features (dict): 特征字典

        Returns:
            tuple: (aigc_score, confidence)
        """
        # 加权特征计算AIGC分数
        weights = {
            # 基础特征权重
            "entropy": -0.2,             # 熵越高，人工创作可能性越大
            "avg_sentence_length": 0.15,  # 句子长度适中的更可能是AI
            "lexical_diversity": -0.3,    # 词汇多样性高，人工创作可能性越大
            "repetition_score": 0.25,     # 重复性越高，AI可能性越大
            "perplexity": -0.2,           # 复杂度越高，人工创作可能性越大
            "function_word_freq": 0.1,    # 功能词频率特征
            "rare_words_ratio": -0.1,     # 罕见词比例越高，人工创作可能性越大
            
            # 高级特征权重(如果存在)
            "readability": 0.15,          # 可读性得分高的更可能是AI
            "sentence_similarity": 0.2,   # 句子相似度高的更可能是AI
            "coherence_score": 0.2,       # 连贯性高的更可能是AI
            "style_consistency": 0.25,    # 风格一致性高的更可能是AI
            "emotion_variation": -0.15,   # 情感变化大的更可能是人工
            "transformers_embedding_1": 0.05,  # transformer特征1
            "transformers_embedding_2": 0.05,  # transformer特征2
            "noun_verb_ratio": 0.1,       # 名词动词比例
            "pos_distribution": -0.15     # 词性分布多样性
        }
        
        # 计算加权分数
        score = 0.5  # 基础分数
        weighted_sum = 0
        total_weight = 0
        
        for feature, weight in weights.items():
            if feature in features:
                # 将特征值归一化到[0,1]范围(假设已经归一化或适用归一化)
                weighted_sum += features[feature] * weight
                total_weight += abs(weight)
        
        if total_weight > 0:
            # 调整基础分数
            score += weighted_sum / total_weight / 2  # 除以2来限制在[0,1]范围
            
        # 确保分数在0到1之间
        score = max(0, min(1, score))
        
        # 置信度计算 (基于特征完整性和一致性)
        feature_coverage = len([f for f in features if f in weights]) / len(weights)
        consistency_score = 0.5 + abs(score - 0.5) * 0.5  # 越接近极端值，一致性越高
        confidence = (feature_coverage * 0.5 + consistency_score * 0.5) 
        
        return score, confidence
        
    def train(self, texts, labels, model_save_path=None, use_advanced_features=True):
        """训练检测器模型(增强版，使用多个模型集成)

        Args:
            texts (list): 文本列表
            labels (list): 标签列表 (1表示AIGC, 0表示人工)
            model_save_path (str, optional): 模型保存路径
            use_advanced_features (bool): 是否使用高级特征

        Returns:
            float: 模型准确率
        """
        if len(texts) < 10:
            logger.warning("训练数据过少，至少需要10个样本")
            return 0.0
            
        # 设置特征提取器的高级特征标志
        self.feature_extractor.use_advanced_features = use_advanced_features
            
        # 提取所有文本的特征
        features_list = []
        for text in texts:
            feature_vector = self.feature_extractor.extract_features(text)
            features_list.append(feature_vector)
        
        # 检查特征维度
        if not features_list:
            logger.error("特征提取失败")
            return 0.0
            
        n_features = len(features_list[0])
        logger.info(f"使用 {n_features} 个特征训练模型")
        
        # 创建模型集合
        models = {
            # 随机森林
            "rf": RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            # 梯度提升树
            "gb": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            # 神经网络
            "nn": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=500,
                random_state=42
            ),
            # 支持向量机
            "svm": SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                random_state=42
            )
        }
        
        # 创建投票分类器(集成模型)
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['rf']),
                ('gb', models['gb']),
                ('nn', models['nn']),
                ('svm', models['svm'])
            ],
            voting='soft'  # 使用概率加权
        )
        
        # 训练所有模型
        X = np.array(features_list)
        y = np.array(labels)
        
        trained_models = {}
        accuracies = {}
        
        # 训练每个单独的模型
        for name, model in models.items():
            try:
                model.fit(X, y)
                acc = model.score(X, y)
                trained_models[name] = model
                accuracies[name] = acc
                logger.info(f"模型 {name} 训练完成，准确率: {acc:.4f}")
            except Exception as e:
                logger.error(f"训练模型 {name} 失败: {str(e)}")
        
        # 训练集成模型
        try:
            ensemble.fit(X, y)
            ensemble_acc = ensemble.score(X, y)
            trained_models['ensemble'] = ensemble
            accuracies['ensemble'] = ensemble_acc
            logger.info(f"集成模型训练完成，准确率: {ensemble_acc:.4f}")
        except Exception as e:
            logger.error(f"训练集成模型失败: {str(e)}")
        
        # 保存所有模型
        self.models = {f"model_{n_features}": trained_models['ensemble']}
        self.models_loaded = True
        
        # 保存模型和元数据
        if model_save_path:
            try:
                # 保存模型
                joblib.dump(self.models, model_save_path)
                
                # 保存模型元数据
                metadata = {
                    "feature_count": n_features,
                    "sample_count": len(texts),
                    "training_accuracies": accuracies,
                    "features_info": {
                        "use_advanced_features": use_advanced_features,
                        "feature_names": list(self.feature_extractor.get_feature_dict(features_list[0]).keys())
                    }
                }
                
                metadata_path = os.path.splitext(model_save_path)[0] + "_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
                    
                logger.info(f"模型保存至: {model_save_path}")
                logger.info(f"模型元数据保存至: {metadata_path}")
            except Exception as e:
                logger.error(f"保存模型失败: {str(e)}")
                
        # 返回集成模型准确率
        return accuracies.get('ensemble', 0.0) 