import React from 'react';
import { Progress, Table, Tag, Tooltip, Typography } from 'antd';

const { Text } = Typography;

const FEATURE_LABELS = {
  char_entropy_norm: '字符熵',
  avg_sentence_length_norm: '平均句长',
  sentence_length_cv_norm: '句长波动',
  lexical_diversity: '词汇多样性',
  hapax_ratio: '一次词比例',
  repetition_ratio: '重复比例',
  bigram_repetition_ratio: '短语重复',
  function_word_ratio: '功能词比例',
  punctuation_ratio: '标点比例',
  long_word_ratio: '长词比例',
  pos_diversity: '词性多样性',
  noun_verb_balance: '名动词平衡',
  assistant_style_cue: '助手话术线索',
  transformer_branch: 'Transformer分支'
};

const DIRECTION_HINT = {
  high: '该特征越高，通常越偏AI',
  low: '该特征越低，通常越偏AI',
  varies: '该特征仅作辅助参考'
};

const getImpactMeta = (row) => {
  if (row.impact === 'increase_ai_risk') {
    return {
      text: '拉高AIGC率',
      color: 'red',
      strokeColor: '#f5222d'
    };
  }
  if (row.impact === 'decrease_ai_risk') {
    return {
      text: '拉低AIGC率',
      color: 'green',
      strokeColor: '#52c41a'
    };
  }
  return {
    text: '中性',
    color: 'default',
    strokeColor: '#1890ff'
  };
};

const FeatureContributionTable = ({ contributions }) => {
  const data = Array.isArray(contributions) ? contributions : [];
  if (!data.length) return null;

  const maxAbsImpact = data.reduce(
    (acc, row) => Math.max(acc, Math.abs(Number(row.total_contribution_percent_points) || 0)),
    1
  );

  const columns = [
    {
      title: '排名',
      dataIndex: 'rank',
      key: 'rank',
      width: 72
    },
    {
      title: '特征',
      dataIndex: 'feature',
      key: 'feature',
      render: (_, record) => {
        const label = FEATURE_LABELS[record.feature] || record.feature;
        const hint = DIRECTION_HINT[record.direction] || DIRECTION_HINT.varies;
        return (
          <Tooltip title={hint}>
            <span>{label}</span>
          </Tooltip>
        );
      }
    },
    {
      title: '原始值',
      dataIndex: 'value_percent',
      key: 'value_percent',
      render: (value) => (
        <Progress
          percent={Number(Number(value || 0).toFixed(1))}
          strokeColor="#1890ff"
          format={(p) => `${Number(p).toFixed(1)}%`}
        />
      )
    },
    {
      title: '总影响(百分点)',
      dataIndex: 'total_contribution_percent_points',
      key: 'total_contribution_percent_points',
      render: (_, record) => {
        const total = Number(record.total_contribution_percent_points || 0);
        const meta = getImpactMeta(record);
        const percent = (Math.abs(total) / maxAbsImpact) * 100;
        return (
          <div>
            <Progress
              percent={Number(percent.toFixed(1))}
              strokeColor={meta.strokeColor}
              format={() => `${total >= 0 ? '+' : ''}${total.toFixed(2)} pp`}
            />
            <Text type="secondary">
              启发式 {record.heuristic_contribution_percent_points >= 0 ? '+' : ''}
              {Number(record.heuristic_contribution_percent_points || 0).toFixed(2)} pp，模型{' '}
              {record.model_contribution_percent_points >= 0 ? '+' : ''}
              {Number(record.model_contribution_percent_points || 0).toFixed(2)} pp
            </Text>
          </div>
        );
      }
    },
    {
      title: '判断',
      dataIndex: 'impact',
      key: 'impact',
      width: 120,
      render: (_, record) => {
        const meta = getImpactMeta(record);
        return <Tag color={meta.color}>{meta.text}</Tag>;
      }
    }
  ];

  return <Table rowKey="feature" columns={columns} dataSource={data} pagination={false} size="small" />;
};

export default FeatureContributionTable;
