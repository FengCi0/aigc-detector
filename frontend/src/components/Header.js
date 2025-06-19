import React from 'react';
import { Layout, Menu } from 'antd';
import { HomeOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { Link, useLocation } from 'react-router-dom';

const { Header: AntHeader } = Layout;

const Header = () => {
  const location = useLocation();
  const currentPath = location.pathname;

  return (
    <AntHeader className="site-header">
      <div className="container">
        <div className="logo">查AIGC率</div>
        <Menu
          theme="light"
          mode="horizontal"
          selectedKeys={[currentPath]}
          style={{ lineHeight: '64px' }}
        >
          <Menu.Item key="/" icon={<HomeOutlined />}>
            <Link to="/">首页</Link>
          </Menu.Item>
          <Menu.Item key="/about" icon={<InfoCircleOutlined />}>
            <Link to="/about">关于</Link>
          </Menu.Item>
        </Menu>
      </div>
    </AntHeader>
  );
};

export default Header; 