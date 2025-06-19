import React from 'react';
import { Table, Progress, Tooltip } from 'antd';

const FeatureTable = ({ features }) => {
  // 特征名称映射和描述
  const featureInfo = {
    entropy: {
      name: '文本熵值',
      description: '衡量文本信息量和随机性，值较低的文本可能更有规律，更可能是AI生成。',
      aiIndicator: 'low' // AI生成内容通常熵值较低
    },
    avg_sentence_length: {
      name: '句子长度',
      description: 'AI生成的内容通常句子长度更为均匀和适中。',
      aiIndicator: 'high' // 高值更可能是AI
    },
    lexical_diversity: {
      name: '词汇多样性',
      description: '词汇的丰富程度，人工撰写内容通常具有更高的词汇多样性。',
      aiIndicator: 'low' // AI生成内容通常词汇多样性较低
    },
    repetition_score: {
      name: '重复度',
      description: '文本中词语和短语的重复程度，AI生成内容通常重复性较高。',
      aiIndicator: 'high' // 高值更可能是AI
    },
    perplexity: {
      name: '复杂度',
      description: '文本结构和内容的复杂性，人工撰写通常具有更高的复杂度。',
      aiIndicator: 'low' // AI生成内容通常复杂度较低
    },
    function_word_freq: {
      name: '功能词频率',
      description: '常见功能词（如"的"、"是"等）的使用频率，AI生成内容可能有特定模式。',
      aiIndicator: 'varies' // 视情况而定
    },
    rare_words_ratio: {
      name: '罕见词比例',
      description: '不常见词汇在文本中的比例，人工撰写通常包含更多罕见词。',
      aiIndicator: 'low' // AI生成内容通常罕见词较少
    },
    // 添加高级特征信息
    readability: {
      name: '可读性',
      description: '文本的易读程度，AI生成内容通常可读性较高。',
      aiIndicator: 'high'
    },
    sentence_similarity: {
      name: '句子相似度',
      description: '句子间的相似程度，AI生成内容句子间相似性通常较高。',
      aiIndicator: 'high'
    },
    coherence_score: {
      name: '连贯性',
      description: '文本的逻辑连贯程度，AI生成内容通常连贯性较高。',
      aiIndicator: 'high'
    },
    style_consistency: {
      name: '风格一致性',
      description: '文本风格的一致程度，AI生成内容通常风格一致性较高。',
      aiIndicator: 'high'
    },
    emotion_variation: {
      name: '情感变化',
      description: '文本中情感表达的变化程度，人工撰写通常情感变化更自然。',
      aiIndicator: 'low'
    },
    transformers_embedding_1: {
      name: '语言模型特征1',
      description: '深度语言模型提取的文本特征。',
      aiIndicator: 'varies'
    },
    transformers_embedding_2: {
      name: '语言模型特征2',
      description: '深度语言模型提取的文本特征。',
      aiIndicator: 'varies'
    },
    noun_verb_ratio: {
      name: '名动词比例',
      description: '名词与动词的比例，AI生成内容可能有特定比例模式。',
      aiIndicator: 'varies'
    },
    pos_distribution: {
      name: '词性分布',
      description: '文本中词性的分布多样性，人工撰写通常词性分布更丰富。',
      aiIndicator: 'low'
    }
  };

  // 获取特征评分
  const getFeatureScore = (feature, value) => {
    // 检查特征是否存在于featureInfo中
    if (!featureInfo[feature]) {
      // 对于未知特征，返回默认评分
      return {
        percent: value * 100,
        strokeColor: '#1890ff'
      };
    }
    
    // 根据特征AI指标方向确定评分颜色
    const percent = value * 100;
    
    if (featureInfo[feature].aiIndicator === 'high') {
      // 高值指向AI
      return {
        percent,
        strokeColor: percent > 70 ? '#f5222d' : percent > 30 ? '#faad14' : '#52c41a'
      };
    } else if (featureInfo[feature].aiIndicator === 'low') {
      // 低值指向AI
      return {
        percent,
        strokeColor: percent < 30 ? '#f5222d' : percent < 70 ? '#faad14' : '#52c41a'
      };
    } else {
      // 中性或不确定
      return {
        percent,
        strokeColor: '#1890ff'
      };
    }
  };

  const columns = [
    {
      title: '特征',
      dataIndex: 'feature',
      key: 'feature',
      render: (text, record) => (
        <Tooltip title={record.description}>
          <span>{text}</span>
        </Tooltip>
      )
    },
    {
      title: '评分',
      dataIndex: 'score',
      key: 'score',
      render: (text, record) => {
        const { percent, strokeColor } = getFeatureScore(record.key, record.rawScore);
        return <Progress percent={percent.toFixed(1)} strokeColor={strokeColor} />;
      }
    },
    {
      title: 'AI指标',
      dataIndex: 'aiIndicator',
      key: 'aiIndicator',
      width: 100,
      render: (text) => {
        if (text === 'high') return <span className="danger-color">较高</span>;
        if (text === 'low') return <span className="success-color">较低</span>;
        return <span className="primary-color">不确定</span>;
      }
    }
  ];

  // 准备表格数据
  const data = Object.keys(features).map(key => {
    // 对于未知特征，提供默认信息
    const featureData = featureInfo[key] || {
      name: key,
      description: '额外检测特征',
      aiIndicator: 'varies'
    };
    
    return {
      key,
      feature: featureData.name,
      description: featureData.description,
      rawScore: features[key],
      aiIndicator: featureData.aiIndicator
    };
  });

  return (
    <Table
      columns={columns}
      dataSource={data}
      pagination={false}
      size="middle"
    />
  );
};

export default FeatureTable;