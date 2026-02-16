# AIGC Detector

中文 AIGC 文本检测项目（发布版），目标是单入口、单数据目录、单生产模型。

[![GitHub stars](https://img.shields.io/github/stars/FengCi0/aigc-detector?style=social)](https://github.com/FengCi0/aigc-detector/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/FengCi0/aigc-detector?style=social)](https://github.com/FengCi0/aigc-detector/network/members)

## 核心设计

- 单入口：只使用 `python run.py`
- 单数据目录：`data/dataset/ai` 与 `data/dataset/human`
- 单生产模型路径：`data/models/aigc_detector_model.joblib`
- 大模型直检：存在可用 Transformer 模型时，服务默认以 Transformer 为主导进行判定

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. 准备数据

将训练文本放入：

- `data/dataset/ai/*.txt`
- `data/dataset/human/*.txt`

### 3. 一键训练（推荐）

```bash
python run.py train-full \
  --ai-data data/dataset/ai \
  --human-data data/dataset/human \
  --base-model hfl/chinese-roberta-wwm-ext \
  --epochs 2 \
  --batch-size 8
```

训练完成后会得到：

- `data/models/aigc_detector_model.joblib`
- `data/models/transformer_detector/`
- `data/models/runtime_config.json`

发布版仓库已内置一个可直接使用的轻量已训练模型：

- `data/models/aigc_detector_model.joblib`
- `data/models/aigc_detector_model.metadata.json`

说明：当前仓库默认不开启 Transformer 大模型权重（体积较大，不随普通 Git 提交）。

### Transformer 模型说明

- 当前仓库默认提供并内置的是轻量已训练模型（可直接运行）。
- Transformer 大模型权重文件通常超过 GitHub 单文件限制（100MB），因此不直接放在仓库中。
- 如需获取作者训练好的 Transformer 权重包或交流部署问题，可联系作者微信：`FengCi_00`

### 4. 启动服务

```bash
python run.py
```

默认地址：`http://127.0.0.1:5000`

如果存在 `frontend/build`，可直接在浏览器访问同端口 UI。

## 公开数据集构建（可选）

可直接构建统一训练集到 `data/dataset`：

```bash
python backend/scripts/prepare_massive_open_mix.py \
  --output-root data/dataset \
  --overwrite
```

支持来源：

- `hc3_zh`（Hello-SimpleAI/HC3-Chinese）
- `hc3_en`（Hello-SimpleAI/HC3）
- `semeval24_mono`（SemEval2024 Task8）
- `daigt`
- `mage`
- `raid`
- `wildchat_zh`

## API

- `GET /health`
- `POST /api/detect`

示例：

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text":"这里是一段足够长的待检测文本，用于验证接口。","include_details":true}'
```

## 测试

```bash
pytest -q
```

## 联系方式

- 微信：`FengCi_00`

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FengCi0/aigc-detector&type=Date)](https://star-history.com/#FengCi0/aigc-detector&Date)
