import re
import math
import numpy as np
from collections import Counter
import jieba
import logging
from scipy.stats import entropy
import nltk
from nltk.tokenize import sent_tokenize
import textstat
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 尝试下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """增强版文本特征提取类，用于AIGC检测"""
    
    def __init__(self, use_advanced_features=True):
        """初始化特征提取器
        
        Args:
            use_advanced_features (bool): 是否使用高级特征（需要额外的模型加载）
        """
        # 中文常用功能词列表（仅示例，实际应更完整）
        self.function_words = set([
            "的", "是", "了", "在", "和", "与", "或", "而", "但", "如果", "虽然", "因为",
            "所以", "那么", "这个", "那个", "这些", "那些", "你", "我", "他", "她", "它",
            "们", "并", "且", "却", "就", "才", "都", "也", "很", "将", "会", "要"
        ])
        
        self.use_advanced_features = use_advanced_features
        self.transformers_model = None
        self.tokenizer = None
        
        # 如果使用高级特征，加载预训练模型
        if use_advanced_features:
            try:
                self._load_pretrained_models()
            except Exception as e:
                logger.warning(f"加载预训练模型失败: {str(e)}，将使用基础特征")
                self.use_advanced_features = False
    
    def _load_pretrained_models(self):
        """加载预训练模型"""
        # 使用缓存目录
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                'data', 'models', 'pretrained')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载BERT tokenizer和模型
        model_name = "bert-base-chinese"  # 使用中文BERT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.transformers_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                num_labels=2
            )
            logger.info(f"成功加载预训练模型: {model_name}")
        except Exception as e:
            logger.error(f"加载预训练模型失败: {str(e)}")
            self.use_advanced_features = False
    
    def extract_features(self, text):
        """提取文本的所有特征

        Args:
            text (str): 输入文本

        Returns:
            list: 特征向量
        """
        if not text or len(text) < 10:
            logger.warning("文本过短，特征提取可能不准确")
            return [0.5] * (7 if not self.use_advanced_features else 15)  # 返回默认特征值
        
        try:
            # 分词
            words = list(jieba.cut(text))
            sentences = re.split(r'[。！？!?]', text)
            sentences = [s for s in sentences if s.strip()]
            
            # 计算基础特征
            entropy_val = self._calculate_entropy(text)
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
            lexical_diversity = len(set(words)) / len(words) if words else 0
            repetition_score = self._calculate_repetition(words)
            perplexity = self._estimate_perplexity(words)
            function_word_freq = self._function_word_frequency(words)
            rare_words_ratio = self._rare_words_ratio(words)
            
            # 基础特征向量
            features = [
                self._normalize_entropy(entropy_val),
                self._normalize_sentence_length(avg_sentence_length),
                lexical_diversity,
                repetition_score,
                self._normalize_perplexity(perplexity),
                function_word_freq,
                rare_words_ratio
            ]
            
            # 如果启用高级特征，计算附加特征
            if self.use_advanced_features:
                # 计算高级文本特征
                readability = self._calculate_readability(text)
                sentence_similarity = self._calculate_sentence_similarity(sentences)
                coherence_score = self._calculate_coherence_score(sentences)
                style_consistency = self._calculate_style_consistency(sentences)
                emotion_variation = self._calculate_emotion_variation(sentences)
                transformers_features = self._get_transformer_embeddings(text)
                noun_verb_ratio = self._calculate_noun_verb_ratio(words)
                pos_distribution = self._calculate_pos_distribution(words)
                
                # 添加高级特征
                features.extend([
                    readability,
                    sentence_similarity,
                    coherence_score,
                    style_consistency,
                    emotion_variation,
                    *transformers_features,  # 扁平化转换器特征
                    noun_verb_ratio,
                    pos_distribution
                ])
            
            return features
        
        except Exception as e:
            logger.error(f"特征提取错误: {str(e)}")
            return [0.5] * (7 if not self.use_advanced_features else 15)  # 返回默认特征值
    
    def get_feature_dict(self, feature_vector):
        """将特征向量转换为具名特征字典

        Args:
            feature_vector (list): 特征向量

        Returns:
            dict: 特征字典
        """
        base_feature_names = [
            "entropy", "avg_sentence_length", "lexical_diversity",
            "repetition_score", "perplexity", "function_word_freq",
            "rare_words_ratio"
        ]
        
        advanced_feature_names = [
            "readability", "sentence_similarity", "coherence_score",
            "style_consistency", "emotion_variation", "transformers_embedding_1",
            "transformers_embedding_2", "noun_verb_ratio", "pos_distribution"
        ]
        
        feature_names = base_feature_names
        if self.use_advanced_features and len(feature_vector) > len(base_feature_names):
            feature_names.extend(advanced_feature_names)
            
        # 截断或扩展feature_names以匹配feature_vector的长度
        if len(feature_names) < len(feature_vector):
            feature_names.extend([f"feature_{i+1}" for i in range(len(feature_names), len(feature_vector))])
        elif len(feature_names) > len(feature_vector):
            feature_names = feature_names[:len(feature_vector)]
            
        return dict(zip(feature_names, feature_vector))
    
    def _calculate_entropy(self, text):
        """计算文本的熵值

        Args:
            text (str): 输入文本

        Returns:
            float: 熵值
        """
        # 计算字符频率
        char_freq = Counter(text)
        total_chars = len(text)
        
        # 计算概率
        probs = [count / total_chars for count in char_freq.values()]
        
        # 计算熵
        return entropy(probs)
    
    def _normalize_entropy(self, entropy_val):
        """归一化熵值到0-1范围"""
        # 基于经验值设定的范围（可根据实际语料调整）
        min_entropy, max_entropy = 3.0, 7.0
        return min(1.0, max(0.0, (entropy_val - min_entropy) / (max_entropy - min_entropy)))
    
    def _normalize_sentence_length(self, length):
        """归一化句子长度到0-1范围"""
        # AI生成的句子长度往往较为规范，人类句子长度差异大
        # 这里设定一个基于经验的理想长度范围，过长或过短都降低分数
        ideal_length = 25  # 假设的理想平均句长（汉语）
        
        # 计算与理想长度的接近程度（越接近越可能是AI）
        distance = abs(length - ideal_length) / ideal_length
        return max(0.0, min(1.0, 1.0 - distance))
    
    def _calculate_repetition(self, words):
        """计算文本重复度

        Args:
            words (list): 分词列表

        Returns:
            float: 重复度得分 (0-1)
        """
        if not words:
            return 0.0
            
        # 计算词频
        word_freq = Counter(words)
        total_words = len(words)
        unique_words = len(word_freq)
        
        # 计算重复词的比例
        repeated_words = sum(freq - 1 for freq in word_freq.values())
        
        # 归一化得分
        if total_words <= 1:
            return 0.0
            
        repetition_ratio = repeated_words / (total_words - 1)  # 减1避免除零
        return min(1.0, repetition_ratio)
    
    def _estimate_perplexity(self, words):
        """估计文本复杂度（简化版perplexity）

        Args:
            words (list): 分词列表

        Returns:
            float: 复杂度估计值
        """
        if not words or len(words) < 2:
            return 0.0
            
        # 计算相邻词的变化程度，作为复杂度的估计
        changes = 0
        for i in range(len(words) - 1):
            if words[i] != words[i+1]:
                changes += 1
        
        return changes / (len(words) - 1)
    
    def _normalize_perplexity(self, perplexity):
        """归一化复杂度分数"""
        # 复杂度高的文本可能更倾向于人工编写
        return perplexity
    
    def _function_word_frequency(self, words):
        """计算功能词频率

        Args:
            words (list): 分词列表

        Returns:
            float: 功能词频率 (0-1)
        """
        if not words:
            return 0.5
            
        # 计算功能词数量
        func_words = sum(1 for word in words if word in self.function_words)
        
        # 计算功能词频率
        return min(1.0, func_words / len(words))
    
    def _rare_words_ratio(self, words):
        """计算罕见词比例（简化版）

        Args:
            words (list): 分词列表

        Returns:
            float: 罕见词比例 (0-1)
        """
        if not words:
            return 0.0
            
        # 仅考虑低频词比例（简化实现）
        word_freq = Counter(words)
        rare_words = sum(1 for word, freq in word_freq.items() if freq == 1 and len(word) > 1)
        
        return min(1.0, rare_words / len(set(words)) if set(words) else 0)
    
    # ===== 高级特征方法 =====
    
    def _calculate_readability(self, text):
        """计算文本可读性得分
        
        AI生成文本通常有特定的可读性模式，使用多种可读性指标组合
        """
        try:
            # 简化计算，基于句子长度和词汇复杂度
            sentences = re.split(r'[。！？!?]', text)
            sentences = [s for s in sentences if s.strip()]
            
            if not sentences:
                return 0.5
                
            # 平均句长
            avg_sent_len = sum(len(s) for s in sentences) / len(sentences)
            
            # 词汇复杂度 (使用多字符词比例)
            words = list(jieba.cut(text))
            complex_words = sum(1 for w in words if len(w) > 2)
            complex_ratio = complex_words / len(words) if words else 0
            
            # 组合得分 (0-1)
            readability = (1 - abs(avg_sent_len - 20) / 40) * 0.5 + complex_ratio * 0.5
            return max(0.0, min(1.0, readability))
        except:
            return 0.5
    
    def _calculate_sentence_similarity(self, sentences):
        """计算句子间的相似度
        
        AI生成文本句子间相似度通常比人工更稳定
        """
        if len(sentences) < 2:
            return 0.5
        
        try:
            # 计算相邻句子间的单词重叠比例
            similarities = []
            for i in range(len(sentences) - 1):
                words1 = set(jieba.cut(sentences[i]))
                words2 = set(jieba.cut(sentences[i + 1]))
                
                if not words1 or not words2:
                    continue
                    
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                
                similarity = len(intersection) / len(union)
                similarities.append(similarity)
            
            # 计算平均相似度和标准差
            if not similarities:
                return 0.5
                
            avg_similarity = sum(similarities) / len(similarities)
            return avg_similarity
        except:
            return 0.5
    
    def _calculate_coherence_score(self, sentences):
        """计算文本连贯性得分
        
        人类vs AI文本的连贯性可能存在差异
        """
        if len(sentences) < 3:
            return 0.5
        
        try:
            # 简化实现，基于句子长度变化的一致性
            sent_lengths = [len(s) for s in sentences]
            length_diffs = [abs(sent_lengths[i] - sent_lengths[i+1]) for i in range(len(sent_lengths)-1)]
            
            # 计算差异的标准差 (标准差越小，一致性越高，可能是AI)
            std_dev = np.std(length_diffs) if length_diffs else 0
            max_std = 30  # 假设的最大标准差
            
            # 归一化，越接近1表示越可能是AI
            coherence = 1.0 - min(std_dev / max_std, 1.0)
            return coherence
        except:
            return 0.5
    
    def _calculate_style_consistency(self, sentences):
        """计算文本风格一致性
        
        AI生成文本通常风格更加一致
        """
        if len(sentences) < 2:
            return 0.5
            
        try:
            # 计算句子结构特征
            punct_ratios = []
            word_length_avgs = []
            
            for sent in sentences:
                # 标点符号比例
                punct_count = len(re.findall(r'[，。！？、：；""]', sent))
                punct_ratio = punct_count / len(sent) if sent else 0
                punct_ratios.append(punct_ratio)
                
                # 平均词长
                words = list(jieba.cut(sent))
                avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
                word_length_avgs.append(avg_word_len)
            
            # 计算特征的变异系数 (变异系数小表示更一致，更可能是AI)
            punct_cv = np.std(punct_ratios) / np.mean(punct_ratios) if np.mean(punct_ratios) else 0
            word_len_cv = np.std(word_length_avgs) / np.mean(word_length_avgs) if np.mean(word_length_avgs) else 0
            
            # 归一化得分 (越接近1越可能是AI)
            max_cv = 1.0
            style_consistency = 1.0 - min((punct_cv + word_len_cv) / 2, max_cv) / max_cv
            return style_consistency
        except:
            return 0.5
    
    def _calculate_emotion_variation(self, sentences):
        """计算情感变化程度
        
        人工文本情感变化可能比AI更自然
        """
        if len(sentences) < 2:
            return 0.5
            
        try:
            # 简化方法，基于感叹词和强调词的使用
            emotion_markers = ['!', '！', '?', '？', '非常', '真的', '太', '好', '坏', '糟糕', '优秀', '可怕', '喜欢', '讨厌']
            emotion_scores = []
            
            for sent in sentences:
                score = 0
                for marker in emotion_markers:
                    if marker in sent:
                        score += 1
                emotion_scores.append(score)
            
            # 计算情感变化
            if not emotion_scores:
                return 0.5
                
            changes = sum(abs(emotion_scores[i] - emotion_scores[i+1]) for i in range(len(emotion_scores)-1))
            max_possible_changes = len(emotion_scores) - 1
            
            # 归一化 (变化多表示可能是人工)
            if max_possible_changes == 0:
                return 0.5
                
            emotion_variation = min(1.0, changes / max_possible_changes)
            return emotion_variation
        except:
            return 0.5
    
    def _get_transformer_embeddings(self, text):
        """获取预训练模型的特征向量
        
        使用Transformer模型提取语义特征
        """
        if not self.use_advanced_features or not self.transformers_model or not self.tokenizer:
            return [0.5, 0.5]  # 默认返回
            
        try:
            # 截断文本以符合模型限制
            max_length = 512  # BERT模型的最大长度
            if len(text) > max_length * 4:  # 如果文本过长
                text = text[:max_length * 4]  # 取前部分
            
            # 使用预训练模型进行特征提取
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
            
            with torch.no_grad():
                outputs = self.transformers_model(**inputs)
            
            # 获取最后一层的隐藏状态
            if hasattr(outputs, 'last_hidden_state'):
                # 取平均并标准化
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                # 从高维向量提取两个有代表性的特征
                if len(embeddings) >= 2:
                    feature1 = np.mean(embeddings[:len(embeddings)//2])
                    feature2 = np.mean(embeddings[len(embeddings)//2:])
                    
                    # 归一化到0-1
                    feature1 = (feature1 + 1) / 2  # Transformer特征通常在-1到1之间
                    feature2 = (feature2 + 1) / 2
                    
                    return [feature1, feature2]
                    
            # 如果前面的方法失败，使用logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits.squeeze().numpy()
                if len(logits) >= 2:
                    # 转换为概率
                    probs = 1 / (1 + np.exp(-logits))  # sigmoid函数
                    return [float(probs[0]), float(probs[1])]
                    
            return [0.5, 0.5]  # 默认返回
        except Exception as e:
            logger.warning(f"Transformer特征提取失败: {str(e)}")
            return [0.5, 0.5]
    
    def _calculate_noun_verb_ratio(self, words):
        """计算名词和动词的比例
        
        通常AI和人类的词性分布有所不同
        """
        if not words:
            return 0.5
            
        try:
            # 简化版，使用词汇前缀估计词性
            noun_prefixes = ['人', '事', '物', '时', '地', '书', '山', '水', '车', '房', '国', '家']
            verb_prefixes = ['说', '做', '看', '走', '想', '吃', '喝', '玩', '写', '读', '学', '教']
            
            nouns = sum(1 for w in words if any(w.startswith(p) for p in noun_prefixes))
            verbs = sum(1 for w in words if any(w.startswith(p) for p in verb_prefixes))
            
            # 避免除以零
            if verbs == 0:
                verbs = 1
                
            # 计算名词/动词比例并归一化
            ratio = nouns / verbs
            norm_ratio = min(1.0, ratio / 3.0)  # 假设最大比例为3
            
            return norm_ratio
        except:
            return 0.5
    
    def _calculate_pos_distribution(self, words):
        """计算词性分布的多样性
        
        通常AI生成文本的词性分布多样性低于人类
        """
        if not words:
            return 0.5
            
        try:
            # 简化版，按词长分类
            length_groups = Counter([len(w) for w in words])
            total_words = len(words)
            
            # 计算分布熵值
            probs = [count / total_words for count in length_groups.values()]
            distribution_entropy = entropy(probs)
            
            # 归一化到0-1
            max_entropy = 3.0  # 假设的最大熵值
            return min(1.0, distribution_entropy / max_entropy)
        except:
            return 0.5 