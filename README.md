# 查AIGC率 - AI生成内容检测工具

这是一个用于检测文本内容是否由AI生成的工具，可以分析文本并给出AIGC生成可能性的评分。

## 功能特点

- 基于多种文本特征分析AIGC生成概率
- **支持预训练语言模型**（BERT）检测
- **集成学习**方法提高检测准确率
- 支持中文文本分析
- 提供API接口，方便集成到其他应用
- 可以训练自定义模型以提高准确率
- 返回详细特征分析，帮助理解检测结果
- **支持长文本分段分析**

## 项目结构

```
aigc-detector/
│
├── backend/                # 后端应用
│   ├── app.py              # 主应用入口
│   ├── run.py              # 启动脚本
│   ├── train_model.py      # 模型训练脚本
│   ├── models/             # AI检测模型
│   │   ├── detector.py     # 检测器实现
│   │   └── features.py     # 特征提取器
│   ├── api/                # API接口
│   │   └── routes.py       # 路由定义
│   └── utils/              # 工具函数
│       └── text_processing.py # 文本处理
│
├── frontend/               # 前端应用
│   ├── public/             # 静态资源
│   └── src/                # 源代码
│       ├── components/     # UI组件
│       ├── pages/          # 页面
│       └── services/       # API调用服务
│
├── data/                   # 数据目录
│   ├── training/           # 训练数据
│   └── models/             # 模型存储
│       └── pretrained/     # 预训练模型缓存
│
└── requirements.txt        # 依赖项
```

## 安装与设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/aigc-detector.git
cd aigc-detector
```

2. 安装依赖：
```bash
pip install -r requirements.txt
cd frontend
npm install
```

3. 运行后端服务：
```bash
cd backend
python run.py
```

4. 运行前端应用（新终端）：
```bash
cd frontend
npm start
```

## 高级特征说明

本工具使用多种特征来检测AI生成内容：

### 基础特征
- **熵值分析**：衡量文本信息量和随机性
- **句子长度分析**：AI生成文本句长分布更稳定
- **词汇多样性**：人类通常使用更多样化的词汇
- **重复度分析**：AI生成内容可能有更多重复模式
- **复杂度评估**：衡量文本结构复杂程度
- **功能词频率**：检测常见功能词使用模式
- **罕见词比例**：人类更可能使用罕见词

### 高级特征
- **可读性评分**：基于多种可读性指标的综合评分
- **句子相似度**：检测句子间的相似程度
- **连贯性分析**：评估文本整体连贯性
- **风格一致性**：检测风格变化
- **情感变化分析**：分析文本中情感表达的变化
- **BERT语义特征**：使用预训练语言模型提取深层语义特征
- **词性分布**：分析名词、动词等词性的分布特点

## 模型训练

可以使用自己的数据集训练模型，提高检测准确率：

```bash
python backend/train_model.py --ai-data data/training/ai_texts --human-data data/training/human_texts --use-advanced
```

参数说明：
- `--ai-data`：AI生成文本目录
- `--human-data`：人工撰写文本目录
- `--output`：模型输出路径(默认`data/models/aigc_detector_model.joblib`)
- `--max-files`：每类最大文件数
- `--min-length`：最小文本长度(默认50)
- `--use-advanced`：使用高级特征和模型

## API使用方法

### 检测文本的AIGC率

**请求：**

```
POST /api/detect
Content-Type: application/json

{
    "text": "需要检测的文本内容..."
}
```

**响应：**

```json
{
    "aigc_score": 75.5,       // AIGC可能性评分（百分比）
    "confidence": 85.2,       // 置信度（百分比）
    "features": {             // 提取的特征值
        "entropy": 0.65,
        "avg_sentence_length": 0.78,
        "lexical_diversity": 0.45,
        "repetition_score": 0.32,
        "perplexity": 0.55,
        "function_word_freq": 0.48,
        "rare_words_ratio": 0.12,
        // 高级特征（若启用）
        "readability": 0.72,
        "sentence_similarity": 0.68,
        "coherence_score": 0.81,
        // ...其他特征
    },
    "processing_time": 0.125  // 处理时间（秒）
}
```

## 未来计划

- 支持更多语言的检测
- 实现更先进的对抗检测方法
- 添加批量处理能力
- 支持更多类型的模型

## 许可证

[MIT 许可证](LICENSE) 