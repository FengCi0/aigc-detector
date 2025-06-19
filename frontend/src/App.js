import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import 'antd/dist/reset.css';
import './App.css';

// 导入组件
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';

const { Content } = Layout;

function App() {
  return (
    <Router>
      <Layout className="layout">
        <Header />
        <Content className="site-content">
          <div className="container">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </div>
        </Content>
        <Footer />
      </Layout>
    </Router>
  );
}

export default App; 