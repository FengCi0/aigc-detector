import React, { useState } from 'react';
import { Card, Typography, Input, Button, Spin, message, Progress, Divider } from 'antd';
import { FileTextOutlined, RobotOutlined, SendOutlined } from '@ant-design/icons';
import FeatureRadarChart from '../components/FeatureRadarChart';
import FeatureTable from '../components/FeatureTable';
import { detectAIGC } from '../services/api';

const { Title, Paragraph } = Typography;
const { TextArea } = Input;

const HomePage = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleSubmit = async () => {
    if (!text.trim()) {
      message.error('请输入需要检测的文本内容');
      return;
    }

    try {
      setLoading(true);
      const data = await detectAIGC(text);
      setResult(data);
      setLoading(false);
    } catch (error) {
      console.error('检测出错:', error);
      message.error('检测失败，请稍后重试');
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score < 30) return '#52c41a'; // 绿色 - 人工可能性高
    if (score < 70) return '#faad14'; // 黄色 - 不确定
    return '#f5222d'; // 红色 - AI可能性高
  };

  const getScoreText = (score) => {
    if (score < 30) return '很可能是人工创作';
    if (score < 70) return '无法确定来源';
    return '很可能是AI生成';
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
          placeholder="请输入需要检测的文本内容（至少50个字）..."
        />
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
                strokeColor={getScoreColor(result.aigc_score)}
                width={120}
              />
              <div style={{ marginTop: 16 }}>
                <Title level={4} style={{ color: getScoreColor(result.aigc_score) }}>
                  {getScoreText(result.aigc_score)}
                </Title>
                <Paragraph>置信度: {result.confidence}%</Paragraph>
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
                <FeatureTable features={result.features} />
              </div>
            </div>
          </Card>
        </Card>
      )}
    </div>
  );
};

export default HomePage; 