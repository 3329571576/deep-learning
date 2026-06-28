# 基于 Bi-LSTM 的中文游戏评论情感分类

深度学习课程设计项目。使用 TapTap 中文游戏评论数据集，构建双向 LSTM 情感分类模型，自动判断评论为「满意」或「不满意」。

## 项目简介

| 项目 | 说明 |
|------|------|
| 任务类型 | 二分类（Binary Classification） |
| 输入 | 中文游戏评论文本 |
| 输出 | `0` = 不满意，`1` = 满意 |
| 模型 | Embedding + Bi-LSTM + Attention Pooling + 全连接 |
| 最佳结果 | 测试集 Accuracy **76.33%**，Macro F1 **76.18%**（Balanced 配置） |

## 数据集

数据来源于 TapTap 平台公开评论（约 300 款游戏，共 4888 条）：

- **review**：评论正文
- **sentiment**：情感标签（评分低于 3 星为 0，其余为 1）

已按 **7 : 1.5 : 1.5** 分层划分为训练集、验证集和测试集：

| 划分 | 样本数 |
|------|--------|
| train | 3421 |
| val | 733 |
| test | 735 |

数据文件位于 `data/` 目录。

## 项目结构

```
Emotional_Learning/
├── data/                      # 划分后的数据集
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── dataset.py                 # 分词、词表、Dataset
├── model.py                   # Bi-LSTM + Attention 模型
├── train.py                   # 训练主脚本
├── split_dataset.py           # 原始数据划分脚本
├── compare_results.py         # 多组实验结果对比
├── 课程设计.ipynb              # 课程设计报告（含可视化与分析）
├── run_train.bat              # 启动 Baseline 训练
├── run_balanced.bat           # 启动 Balanced 训练（推荐）
├── run_optimized.bat          # 启动 Optimized 训练
├── checkpoints/               # Baseline 实验结果
├── checkpoints_balanced/      # Balanced 实验结果（最终模型）
├── checkpoints_optimized/     # Optimized 实验结果
└── requirements.txt           # Python 依赖
```

## 环境配置

推荐使用 conda 环境 **test4**（Python 3.12.9）：

```bash
conda activate test4
pip install -r requirements.txt
```

主要依赖：`torch`、`pandas`、`jieba`、`scikit-learn`、`matplotlib`、`seaborn`

## 快速开始

### 1. 数据划分（可选）

若已有 `data/` 目录下的 CSV，可跳过此步。

```bash
python split_dataset.py
```

### 2. 模型训练

```bash
# Baseline：原始 Bi-LSTM 配置
python train.py --profile baseline

# Balanced：折中配置（推荐，性能最佳）
python train.py --profile balanced

# Optimized：强正则化配置
python train.py --profile optimized
```

也可直接运行批处理脚本：

```bash
run_balanced.bat
```

### 3. 对比实验结果

```bash
python compare_results.py
```

### 4. 查看课程设计报告

在 Jupyter / VS Code 中打开 `课程设计.ipynb`，运行全部单元格即可生成数据分析、实验对比与可视化图表。

## 实验配置与结果

三组 Bi-LSTM 超参数配置对比如下：

| 超参数 | Baseline | Optimized | Balanced |
|--------|----------|-----------|----------|
| max_len | 256 | 128 | 256 |
| hidden_dim | 128 | 96 | 128 |
| num_layers | 2 | 1 | 2 |
| pooling | last | attention | attention |
| dropout | 0.5 | 0.6 | 0.5 |
| lr | 1e-3 | 5e-4 | 7e-4 |
| lr_scheduler | 无 | plateau | plateau |

测试集性能对比：

| 模型 | Accuracy | Macro F1 |
|------|----------|----------|
| Bi-LSTM Baseline | 73.06% | 72.99% |
| Bi-LSTM Optimized | 71.29% | 71.10% |
| **Bi-LSTM Balanced** | **76.33%** | **76.18%** |

**结论**：Balanced 配置在保留模型容量与完整序列长度的同时，引入注意力池化与适度正则化，取得最佳测试性能，作为最终选用模型。

训练完成后，各实验目录下会生成：

| 文件 | 说明 |
|------|------|
| `best_model.pt` | 最佳模型权重 |
| `vocab.pkl` | 词表 |
| `config.json` | 超参数配置 |
| `history.json` | 每轮训练指标 |
| `test_result.json` | 测试集最终结果 |

## 模型架构

```
输入评论
  → jieba 分词 → 词 ID 序列
  → Embedding（128 维）
  → 双向 LSTM（2 层，hidden=128）
  → Attention 加权池化
  → Dropout → 全连接（2 类）
  → Softmax → 输出（0 / 1）
```

## 常用训练参数

```bash
python train.py --profile balanced --epochs 30 --batch-size 32 --lr 5e-4
python train.py --profile custom --hidden-dim 128 --dropout 0.5 --pooling attention
```

完整参数说明：

```bash
python train.py --help
```

## 预处理流程

1. jieba 中文分词
2. 在训练集上构建词表（低频词过滤，UNK 映射）
3. 序列截断至 `max_len`，batch 内 padding
4. 使用 `pack_padded_sequence` 处理变长序列

## 许可证

数据集原始说明见项目内 `dataset_infos.json`。TapTap 评论数据集用于 NLP 研究与课程学习。
