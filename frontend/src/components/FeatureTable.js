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
    }
  };

  // 获取特征评分
  const getFeatureScore = (feature, value) => {
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
  const data = Object.keys(features).map(key => ({
    key,
    feature: featureInfo[key]?.name || key,
    description: featureInfo[key]?.description || '',
    rawScore: features[key],
    aiIndicator: featureInfo[key]?.aiIndicator || 'varies'
  }));

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