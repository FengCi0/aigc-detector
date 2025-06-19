import React from 'react';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js';

// 注册ChartJS组件
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const FeatureRadarChart = ({ features }) => {
  // 特征展示名称映射
  const featureLabels = {
    // 基础特征
    entropy: '文本熵值',
    avg_sentence_length: '句子长度',
    lexical_diversity: '词汇多样性',
    repetition_score: '重复度',
    perplexity: '复杂度',
    function_word_freq: '功能词频率',
    rare_words_ratio: '罕见词比例',
    
    // 高级特征
    readability: '可读性',
    sentence_similarity: '句子相似度',
    coherence_score: '连贯性',
    style_consistency: '风格一致性',
    emotion_variation: '情感变化',
    transformers_embedding_1: '语言模型特征1',
    transformers_embedding_2: '语言模型特征2',
    noun_verb_ratio: '名动词比例',
    pos_distribution: '词性分布'
  };

  // 选择最重要的特征（最多8个）展示在雷达图上
  const priorityFeatures = [
    'entropy', 'lexical_diversity', 'repetition_score', 'perplexity',
    'style_consistency', 'coherence_score', 'sentence_similarity', 'rare_words_ratio'
  ];
  
  // 从features中筛选出最重要的特征
  const selectedFeatures = {};
  
  // 先添加优先特征（如果存在）
  priorityFeatures.forEach(key => {
    if (features[key] !== undefined) {
      selectedFeatures[key] = features[key];
    }
  });
  
  // 如果特征不足8个，添加其他可用特征
  if (Object.keys(selectedFeatures).length < 8) {
    Object.keys(features).forEach(key => {
      if (!selectedFeatures[key] && Object.keys(selectedFeatures).length < 8) {
        selectedFeatures[key] = features[key];
      }
    });
  }

  // 准备雷达图数据
  const data = {
    labels: Object.keys(selectedFeatures).map(key => featureLabels[key] || key),
    datasets: [
      {
        label: 'AIGC特征评分',
        data: Object.values(selectedFeatures).map(val => val * 100), // 转换为百分比
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(54, 162, 235, 1)'
      }
    ]
  };

  // 雷达图配置
  const options = {
    scales: {
      r: {
        min: 0,
        max: 100,
        ticks: {
          stepSize: 20
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.formattedValue}%`;
          }
        }
      }
    },
    maintainAspectRatio: false
  };

  return (
    <div style={{ height: '300px', width: '100%' }}>
      <Radar data={data} options={options} />
    </div>
  );
};

export default FeatureRadarChart; 