import axios from 'axios';

// 创建axios实例
const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || '',
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * 检测文本的AIGC率
 * @param {string} text - 需要检测的文本
 * @param {boolean} includeDetails - 是否返回详细结果
 * @returns {Promise} - 返回检测结果
 */
export const detectAIGC = async (text, includeDetails = false) => {
  try {
    const response = await api.post('/api/detect', { text, include_details: includeDetails });
    return response.data;
  } catch (error) {
    console.error('API请求失败:', error);
    const message =
      error?.response?.data?.error?.message ||
      error?.response?.data?.message ||
      '请求失败，请稍后重试';
    throw new Error(message);
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
