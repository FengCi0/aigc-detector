import React from 'react';
import { Layout } from 'antd';

const { Footer: AntFooter } = Layout;

const Footer = () => {
  return (
    <AntFooter className="site-footer">
      <div className="container">
        查AIGC率 - AI生成内容检测工具 &copy; {new Date().getFullYear()}
      </div>
    </AntFooter>
  );
};

export default Footer; 