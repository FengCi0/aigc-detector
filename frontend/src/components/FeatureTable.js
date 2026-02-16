import React, { useState } from 'react';
import { Progress, Segmented, Space, Table, Tooltip } from 'antd';

const FeatureTable = ({ features }) => {
  const [scoreView, setScoreView] = useState('risk');
  const [sortMode, setSortMode] = useState('risk');

  const featureInfo = {
    char_entropy_norm: {
      name: '字符熵',
      description: '文本字符分布复杂度。通常更高的熵更接近人工写作。',
      aiIndicator: 'low'
    },
    avg_sentence_length_norm: {
      name: '平均句长',
      description: '句子平均长度的归一化值。',
      aiIndicator: 'varies'
    },
    sentence_length_cv_norm: {
      name: '句长波动',
      description: '句子长度波动系数，越低通常表示更模板化。',
      aiIndicator: 'low'
    },
    lexical_diversity: {
      name: '词汇多样性',
      description: '独特词占比。通常越高越接近人工写作。',
      aiIndicator: 'low'
    },
    hapax_ratio: {
      name: '一次词比例',
      description: '只出现一次的词占比，越高通常信息密度更高。',
      aiIndicator: 'low'
    },
    repetition_ratio: {
      name: '重复比例',
      description: '词重复程度，越高通常越模板化。',
      aiIndicator: 'high'
    },
    bigram_repetition_ratio: {
      name: '短语重复',
      description: '相邻词组重复比例，越高通常越模板化。',
      aiIndicator: 'high'
    },
    function_word_ratio: {
      name: '功能词比例',
      description: '常见虚词比例。',
      aiIndicator: 'varies'
    },
    punctuation_ratio: {
      name: '标点比例',
      description: '标点在文本中的占比。',
      aiIndicator: 'varies'
    },
    long_word_ratio: {
      name: '长词比例',
      description: '长度较长词语占比。',
      aiIndicator: 'low'
    },
    pos_diversity: {
      name: '词性多样性',
      description: '词性分布熵，越高通常越接近人工写作。',
      aiIndicator: 'low'
    },
    noun_verb_balance: {
      name: '名动词平衡',
      description: '名词与动词的平衡比例。',
      aiIndicator: 'varies'
    }
  };

  const normalizeValue = (value) => {
    const num = Number(value);
    if (Number.isNaN(num)) return 0.5;
    return Math.min(1, Math.max(0, num));
  };

  const getAiRiskMeta = (feature, value) => {
    const normalized = normalizeValue(value);
    const aiIndicator = featureInfo[feature]?.aiIndicator || 'varies';
    const directionHint =
      aiIndicator === 'high'
        ? '该特征值越高，通常越偏向AI写作。'
        : aiIndicator === 'low'
          ? '该特征值越低，通常越偏向AI写作。'
          : '该特征受语域影响较大，不单独决定AI风险。';

    if (aiIndicator === 'varies') {
      return {
        isReference: true,
        directionHint
      };
    }

    const riskValue = aiIndicator === 'high' ? normalized : 1 - normalized;
    const riskPercent = riskValue * 100;
    const thresholdHint = '阈值：>=75% 高风险，45%-75% 中风险，<45% 低风险。';

    if (riskValue >= 0.75) {
      return {
        isReference: false,
        riskPercent,
        text: 'AI风险高',
        className: 'danger-color',
        strokeColor: '#f5222d',
        directionHint,
        thresholdHint
      };
    }
    if (riskValue >= 0.45) {
      return {
        isReference: false,
        riskPercent,
        text: 'AI风险中',
        className: 'warning-color',
        strokeColor: '#faad14',
        directionHint,
        thresholdHint
      };
    }
    return {
      isReference: false,
      riskPercent,
      text: 'AI风险低',
      className: 'success-color',
      strokeColor: '#52c41a',
      directionHint,
      thresholdHint
    };
  };

  const renderIndicator = (risk) => {
    if (risk.isReference) {
      return (
        <Tooltip title="该特征不单独决定AI风险，仅作参考">
          <span className="primary-color">参考（不参与风险阈值）</span>
        </Tooltip>
      );
    }

    return (
      <Tooltip
        title={
          <div>
            <div>{risk.directionHint}</div>
            <div>{risk.thresholdHint}</div>
          </div>
        }
      >
        <span className={risk.className}>
          {risk.text}（{risk.riskPercent.toFixed(1)}%）
        </span>
      </Tooltip>
    );
  };

  const dataSource = Object.entries(features || {}).map(([key, value]) => {
    const info = featureInfo[key] || {
      name: key,
      description: '额外特征',
      aiIndicator: 'varies'
    };
    const rawPercent = normalizeValue(value) * 100;
    const risk = getAiRiskMeta(key, value);

    return {
      key,
      feature: info.name,
      description: info.description,
      rawScore: value,
      rawPercent,
      risk
    };
  });

  if (sortMode === 'risk') {
    dataSource.sort((a, b) => {
      if (a.risk.isReference && b.risk.isReference) return a.feature.localeCompare(b.feature, 'zh-CN');
      if (a.risk.isReference) return 1;
      if (b.risk.isReference) return -1;
      return b.risk.riskPercent - a.risk.riskPercent;
    });
  } else {
    dataSource.sort((a, b) => a.feature.localeCompare(b.feature, 'zh-CN'));
  }

  const columns = [
    {
      title: '特征',
      dataIndex: 'feature',
      key: 'feature',
      render: (text, record) => (
        <Tooltip
          title={
            <div>
              <div>{record.description}</div>
              <div>{record.risk.directionHint}</div>
            </div>
          }
        >
          <span>{text}</span>
        </Tooltip>
      )
    },
    {
      title: scoreView === 'raw' ? '评分（原始值）' : '评分（AI风险值）',
      dataIndex: 'score',
      key: 'score',
      render: (_, record) => {
        if (scoreView === 'risk') {
          if (record.risk.isReference) {
            return (
              <Tooltip title="该特征是参考项，显示的是原始值，不参与高/中/低风险阈值判定。">
                <Progress
                  percent={Number(record.rawPercent.toFixed(1))}
                  strokeColor="#91caff"
                  format={(p) => `${Number(p).toFixed(1)}%（参考）`}
                />
              </Tooltip>
            );
          }
          return (
            <Progress
              percent={Number(record.risk.riskPercent.toFixed(1))}
              strokeColor={record.risk.strokeColor}
              format={(p) => `${Number(p).toFixed(1)}%`}
            />
          );
        }

        return (
          <Progress
            percent={Number(record.rawPercent.toFixed(1))}
            strokeColor="#1890ff"
            format={(p) => `${Number(p).toFixed(1)}%`}
          />
        );
      }
    },
    {
      title: 'AI指示',
      dataIndex: 'aiIndicator',
      key: 'aiIndicator',
      width: 180,
      render: (_, record) => renderIndicator(record.risk)
    }
  ];

  return (
    <>
      <Space wrap style={{ marginBottom: 12 }}>
        <span>评分视图</span>
        <Segmented
          size="small"
          value={scoreView}
          onChange={setScoreView}
          options={[
            { label: '原始值', value: 'raw' },
            { label: 'AI风险值', value: 'risk' }
          ]}
        />
        <span>特征排序</span>
        <Segmented
          size="small"
          value={sortMode}
          onChange={setSortMode}
          options={[
            { label: '风险优先', value: 'risk' },
            { label: '名称顺序', value: 'name' }
          ]}
        />
      </Space>
      <Table columns={columns} dataSource={dataSource} pagination={false} size="middle" rowKey="key" />
    </>
  );
};

export default FeatureTable;
