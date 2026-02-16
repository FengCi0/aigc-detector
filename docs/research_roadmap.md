# Research Roadmap

## 当前产品目标

1. 单数据目录：`data/dataset/ai` 与 `data/dataset/human`
2. 单入口：`python run.py`
3. 大模型主导检测：Transformer 分支作为主判定器，轻量模型用于鲁棒补充

## 评估闭环

1. 训练：`python run.py train-full --ai-data data/dataset/ai --human-data data/dataset/human`
2. 评估：`python backend/scripts/evaluate_detector.py --model-path data/models/aigc_detector_model.joblib --ai-dir data/dataset/ai --human-dir data/dataset/human`
3. 关注指标：F1、ROC-AUC、PR-AUC、ECE、Brier、LogLoss

## 公开基准建议

- HC3 / HC3-Chinese
- SemEval-2024 Task8
- DAIGT / MAGE / RAID

## 关键论文

- DetectGPT: https://arxiv.org/abs/2301.11305
- Ghostbuster: https://arxiv.org/abs/2305.15047
- RAID: https://aclanthology.org/2024.acl-long.674/
- SemEval-2024 Task8: https://aclanthology.org/2024.semeval-1.279/
