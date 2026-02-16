# AIGC Detector

面向中文场景的 AIGC 文本检测项目，提供 **Web 页面 + API 接口 + 可训练模型**，可直接本地部署使用。

[![GitHub stars](https://img.shields.io/github/stars/FengCi0/aigc-detector?style=social)](https://github.com/FengCi0/aigc-detector/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/FengCi0/aigc-detector?style=social)](https://github.com/FengCi0/aigc-detector/network/members)

## 功能亮点

- 支持中文文本 AIGC 检测，返回概率、标签和可解释特征
- 提供前端页面，可直接在浏览器中粘贴文本检测
- 提供后端 API，便于接入业务系统或二次开发
- 支持本地增量训练，方便根据你的数据持续优化模型

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2) 启动服务

```bash
python run.py
```

默认地址：`http://127.0.0.1:5000`

如果存在 `frontend/build`，可直接在同端口访问 Web UI。

### 3) 接口调用示例

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"这里是一段足够长的待检测文本，用于验证接口。","include_details":true}'
```

## 内置模型说明

仓库已内置可直接使用的轻量已训练模型：

- `data/models/aigc_detector_model.joblib`
- `data/models/aigc_detector_model.metadata.json`

默认可直接启动和检测，无需先训练。

## 训练你自己的模型

### 数据目录格式

将训练文本放入：

- `data/dataset/ai/*.txt`
- `data/dataset/human/*.txt`

建议 `ai` 与 `human` 数量尽量平衡，文本长度尽量不低于 50 字。

### 仅训练轻量模型

```bash
python run.py train \
  --ai-data data/dataset/ai \
  --human-data data/dataset/human
```

### 一键训练完整流程（轻量 + Transformer + 融合配置）

```bash
python run.py train-full \
  --ai-data data/dataset/ai \
  --human-data data/dataset/human \
  --base-model hfl/chinese-roberta-wwm-ext \
  --epochs 2 \
  --batch-size 8
```

## Transformer 模型说明

- Transformer 权重文件通常较大（常超过 GitHub 单文件 100MB 限制），因此默认不随仓库完整发布。
- 当前仓库默认可直接使用内置轻量模型。
- 如需作者训练好的 Transformer 权重包或部署交流，可联系：`FengCi_00`（微信）

## 公开数据集构建（可选）

可使用脚本快速构建训练集到 `data/dataset`：

```bash
python backend/scripts/prepare_massive_open_mix.py \
  --output-root data/dataset \
  --overwrite
```

支持来源包括：`HC3`、`SemEval2024`、`DAIGT`、`MAGE`、`RAID`、`wildchat_zh` 等。

## API

- 健康检查：`GET /health`
- 检测接口：`POST /api/detect`

## 测试

```bash
pytest -q
```

## 联系方式

- 微信：`FengCi_00`

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FengCi0/aigc-detector&type=Date)](https://star-history.com/#FengCi0/aigc-detector&Date)
