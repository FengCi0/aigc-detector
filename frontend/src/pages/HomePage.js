import React, { useEffect, useState } from 'react';
import { Card, Typography, Input, Button, Spin, message, Progress, Divider, Alert, Tag } from 'antd';
import { FileTextOutlined, RobotOutlined, SendOutlined } from '@ant-design/icons';
import FeatureRadarChart from '../components/FeatureRadarChart';
import FeatureContributionTable from '../components/FeatureContributionTable';
import FeatureTable from '../components/FeatureTable';
import { checkHealth, detectAIGC } from '../services/api';

const { Title, Paragraph } = Typography;
const { TextArea } = Input;
const HomePage = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [minLength, setMinLength] = useState(50);

  useEffect(() => {
    let mounted = true;
    checkHealth()
      .then((health) => {
        const min = Number(health?.min_text_length);
        if (mounted && Number.isFinite(min) && min >= 10) {
          setMinLength(min);
        }
      })
      .catch(() => {
        // 忽略健康检查失败，使用本地默认值
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleSubmit = async () => {
    const cleaned = text.trim();
    if (!cleaned) {
      message.error('请输入需要检测的文本内容');
      return;
    }
    if (cleaned.length < minLength) {
      message.error(`文本长度不足，请至少输入${minLength}个字符`);
      return;
    }

    try {
      setLoading(true);
      const data = await detectAIGC(cleaned, true);
      setResult(data);
    } catch (error) {
      console.error('检测出错:', error);
      message.error(error.message || '检测失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score, label, threshold) => {
    const margin = Number(score) - Number(threshold);
    if (label === 'ai') {
      return margin >= 10 ? '#f5222d' : '#faad14';
    }
    return margin <= -10 ? '#52c41a' : '#faad14';
  };

  const getScoreText = (score, label, threshold) => {
    const margin = Number(score) - Number(threshold);
    if (label === 'ai') {
      return margin >= 10 ? '判定：AI生成倾向高' : '判定：AI生成（边界区间）';
    }
    return margin <= -10 ? '判定：人工创作倾向高' : '判定：人工（边界区间）';
  };

  return (
    <div>
      <Title className="site-title">查AIGC率 - AI生成内容检测工具</Title>
      <Paragraph className="text-center mb-20">
        输入文本，快速检测内容是否由AI生成，获取详细分析报告
      </Paragraph>

      <Card className="mb-20">
        <TextArea
          rows={10}
          value={text}
          onChange={handleTextChange}
          placeholder={`请输入需要检测的文本内容（至少${minLength}个字）...`}
        />
        <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
          当前长度：{text.trim().length} 字符，建议输入 200 字以上以提升稳定性
        </Paragraph>
        <Button
          type="primary"
          icon={<SendOutlined />}
          loading={loading}
          onClick={handleSubmit}
          style={{ marginTop: 16, float: 'right' }}
        >
          开始检测
        </Button>
        <div style={{ clear: 'both' }}></div>
      </Card>

      {loading && (
        <div className="text-center mt-20">
          <Spin size="large" />
          <Paragraph style={{ marginTop: 16 }}>正在分析文本，请稍候...</Paragraph>
        </div>
      )}

      {result && !loading && (
        <Card className="result-card">
          <div className="result-header">
            <Title level={3}>检测结果</Title>
            <Paragraph>处理时间: {result.processing_time}秒</Paragraph>
          </div>

          {Array.isArray(result.warnings) && result.warnings.length > 0 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message={result.warnings.join(' ')}
            />
          )}

          <Card
            type="inner"
            title={
              <span>
                <RobotOutlined /> AIGC率评分
              </span>
            }
          >
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <Progress
                type="circle"
                percent={result.aigc_score}
                format={(percent) => `${percent}%`}
                strokeColor={getScoreColor(result.aigc_score, result.label, result.score_threshold)}
                width={120}
              />
              <div style={{ marginTop: 16 }}>
                <Title
                  level={4}
                  style={{ color: getScoreColor(result.aigc_score, result.label, result.score_threshold) }}
                >
                  {getScoreText(result.aigc_score, result.label, result.score_threshold)}
                </Title>
                <Paragraph>置信度: {result.confidence}%</Paragraph>
                <Paragraph type="secondary" style={{ marginTop: 0, marginBottom: 8 }}>
                  当前判定阈值: {result.score_threshold}%
                </Paragraph>
                {result.model_mode && <Tag color="blue">模式: {result.model_mode}</Tag>}
                {typeof result?.details?.transformer_branch_contribution_percent_points === 'number' &&
                  Math.abs(result.details.transformer_branch_contribution_percent_points) > 0.01 && (
                    <Tag color="purple">
                      Transformer贡献: {result.details.transformer_branch_contribution_percent_points >= 0 ? '+' : ''}
                      {result.details.transformer_branch_contribution_percent_points.toFixed(2)} pp
                    </Tag>
                  )}
                {result?.details?.calibration_enabled && (
                  <Tag color="geekblue">
                    概率校准: {result.details.calibration_method || 'enabled'}
                  </Tag>
                )}
                {typeof result?.details?.aigc_probability_raw === 'number' &&
                  typeof result?.details?.score_clip_eps === 'number' &&
                  result.details.score_clip_eps > 0 && (
                    <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
                      原始概率: {(result.details.aigc_probability_raw * 100).toFixed(2)}%，显示概率做了
                      {(result.details.score_clip_eps * 100).toFixed(2)}% 极值抑制以避免绝对化误导。
                    </Paragraph>
                  )}
                {result?.details?.calibration_enabled && result?.details?.calibration_metrics && (
                  <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
                    ECE（验证集）:
                    {` ${(Number(result.details.calibration_metrics.val_ece_before || 0) * 100).toFixed(2)}% -> ${(Number(
                      result.details.calibration_metrics.val_ece_after || 0
                    ) * 100).toFixed(2)}%`}
                    {`，Brier: ${Number(result.details.calibration_metrics.val_brier_before || 0).toFixed(4)} -> ${Number(
                      result.details.calibration_metrics.val_brier_after || 0
                    ).toFixed(4)}`}
                    {`，LogLoss: ${Number(result.details.calibration_metrics.val_log_loss_before || 0).toFixed(
                      4
                    )} -> ${Number(result.details.calibration_metrics.val_log_loss_after || 0).toFixed(4)}`}
                  </Paragraph>
                )}
              </div>
            </div>
          </Card>

          <Divider />

          <Card
            type="inner"
            title={
              <span>
                <FileTextOutlined /> 文本特征分析
              </span>
            }
          >
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between' }}>
              <div style={{ width: '100%', maxWidth: '500px', margin: '0 auto' }}>
                <FeatureRadarChart features={result.features} />
              </div>
              <div style={{ width: '100%', marginTop: 16 }}>
                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message="说明：评分视图可切换“原始值/AI风险值”；AI指示是按特征方向映射后的风险，不等于原始值大小。"
                />
                <FeatureTable features={result.features} />
              </div>
              {Array.isArray(result?.details?.feature_contributions) &&
                result.details.feature_contributions.length > 0 && (
                  <div style={{ width: '100%', marginTop: 20 }}>
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginBottom: 12 }}
                      message="关键影响因子：展示每个特征对最终AIGC率的拉升/拉低贡献（单位：百分点pp）。"
                    />
                    <FeatureContributionTable contributions={result.details.feature_contributions} />
                  </div>
                )}
            </div>
          </Card>
        </Card>
      )}
    </div>
  );
};

export default HomePage; 
