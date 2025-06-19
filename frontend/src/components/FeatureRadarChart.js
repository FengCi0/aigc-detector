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
    entropy: '文本熵值',
    avg_sentence_length: '句子长度',
    lexical_diversity: '词汇多样性',
    repetition_score: '重复度',
    perplexity: '复杂度',
    function_word_freq: '功能词频率',
    rare_words_ratio: '罕见词比例'
  };

  // 准备雷达图数据
  const data = {
    labels: Object.keys(features).map(key => featureLabels[key] || key),
    datasets: [
      {
        label: 'AIGC特征评分',
        data: Object.values(features).map(val => val * 100), // 转换为百分比
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