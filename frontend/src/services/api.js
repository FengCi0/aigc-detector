import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5000',
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * 检测文本的AIGC率
 * @param {string} text - 需要检测的文本
 * @returns {Promise} - 返回检测结果
 */
export const detectAIGC = async (text) => {
  try {
    const response = await api.post('/api/detect', { text });
    return response.data;
  } catch (error) {
    console.error('API请求失败:', error);
    throw error;
  }
};

/**
 * 获取服务器健康状态
 * @returns {Promise} - 返回健康状态
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('健康检查失败:', error);
    throw error;
  }
};

export default api; 