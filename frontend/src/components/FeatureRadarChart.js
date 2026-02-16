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
  const normalizeValue = (value) => {
    const num = Number(value);
    if (Number.isNaN(num)) return 0.5;
    return Math.min(1, Math.max(0, num));
  };

  // 特征展示名称映射
  const featureLabels = {
    // 基础特征
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
    noun_verb_balance: '名动词平衡'
  };

  // 选择最重要的特征（最多8个）展示在雷达图上
  const priorityFeatures = [
    'char_entropy_norm', 'lexical_diversity', 'repetition_ratio', 'bigram_repetition_ratio',
    'sentence_length_cv_norm', 'function_word_ratio', 'pos_diversity', 'noun_verb_balance'
  ];
  
  // 从features中筛选出最重要的特征
  const selectedFeatures = {};
  
  // 确保features是有效对象
  if (!features || typeof features !== 'object') {
    // 如果features无效，返回空对象
    console.warn('无效的特征数据');
  } else {
    // 先添加优先特征（如果存在）
    priorityFeatures.forEach(key => {
      if (features[key] !== undefined) {
        selectedFeatures[key] = features[key];
      }
    });
    
    // 如果特征不足8个，添加其他可用特征
    if (Object.keys(selectedFeatures).length < 8) {
      Object.keys(features).forEach(key => {
        if (selectedFeatures[key] === undefined && Object.keys(selectedFeatures).length < 8) {
          selectedFeatures[key] = features[key];
        }
      });
    }
  }

  // 准备雷达图数据
  const data = {
    labels: Object.keys(selectedFeatures).map(key => featureLabels[key] || key),
    datasets: [
      {
        label: 'AIGC特征评分',
        data: Object.values(selectedFeatures).map(val => normalizeValue(val) * 100), // 转换为百分比
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
