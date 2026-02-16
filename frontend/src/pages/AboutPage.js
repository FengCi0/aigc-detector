import React from 'react';
import { Card, Typography, Divider, List } from 'antd';
import { ExperimentOutlined, BulbOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

const AboutPage = () => {
  const features = [
    '字符熵与词汇多样性 - 评估文本信息密度和丰富性',
    '句长与波动性分析 - 识别模板化表达模式',
    '重复短语检测 - 识别高频复用片段',
    '词性分布与名动词平衡 - 评估语言结构变化',
    '可训练分类模型 + 启发式兜底 - 提升稳定性',
    '长文本分段聚合 - 支持超长输入的稳定分析'
  ];

  const faqs = [
    {
      question: '什么是AIGC率？',
      answer: 'AIGC率是指文本由人工智能生成的可能性百分比。评分越高，表示文本更可能是由AI生成而非人工撰写。'
    },
    {
      question: '检测结果准确吗？',
      answer: '检测结果是概率估计，不是最终判定。即便模型命中率较高，仍可能误判，建议结合来源、写作过程和其他证据综合判断。'
    },
    {
      question: '什么样的文本适合检测？',
      answer: '任何中文文本都可以检测，但文本长度至少需要50个字以上才能获得较准确的结果。学术论文、新闻、文章等均可检测。'
    },
    {
      question: '如何提高检测准确率？',
      answer: '提供较长的文本样本（200字以上）会提高检测准确率。同时，我们持续改进算法和收集更多训练数据。'
    },
    {
      question: '为什么需要检测AI生成内容？',
      answer: '随着AI生成技术的发展，区分人工与AI创作内容变得越来越重要，特别是在学术、新闻和创意领域，确保内容的真实性和原创性。'
    }
  ];

  return (
    <div>
      <Title className="site-title">关于 查AIGC率</Title>

      <Card className="mb-20">
        <Paragraph>
          查AIGC率是一款专注于检测文本是否由AI生成的工具。随着ChatGPT等AI语言模型的普及，
          区分人工创作和AI生成的内容变得越来越重要。我们的工具基于可解释文本特征和可训练模型，
          帮助用户识别潜在的AI生成内容。
        </Paragraph>
      </Card>

      <Card
        className="mb-20"
        title={
          <span>
            <ExperimentOutlined /> 技术特点
          </span>
        }
      >
        <List
          dataSource={features}
          renderItem={(item) => <List.Item>{item}</List.Item>}
        />
      </Card>

      <Card
        title={
          <span>
            <QuestionCircleOutlined /> 常见问题
          </span>
        }
      >
        {faqs.map((faq, index) => (
          <div key={index}>
            <Title level={5}>{faq.question}</Title>
            <Paragraph>{faq.answer}</Paragraph>
            {index < faqs.length - 1 && <Divider />}
          </div>
        ))}
      </Card>

      <Divider />

      <Card
        title={
          <span>
            <BulbOutlined /> 未来计划
          </span>
        }
      >
        <Paragraph>
          我们正在努力改进检测算法，未来将引入以下功能：
        </Paragraph>
        <List
          dataSource={[
            '深度学习模型支持，提高检测准确率',
            '多语言支持，包括英文等其他语言',
            '批量文件分析功能',
            '更详细的文本特征报告',
            'API接口，方便第三方集成'
          ]}
          renderItem={(item) => <List.Item>{item}</List.Item>}
        />
      </Card>
    </div>
  );
};

export default AboutPage; 
