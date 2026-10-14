# 实验一：文本分类（新闻文本 10 分类）

## 环境要求

- Python 3.10（conda 环境 `project1`）
- 依赖库：`numpy`、`pandas`、`scikit-learn`、`matplotlib`

创建环境命令（如未创建）：

```bash
conda create -n project1 python=3.10
conda activate project1
conda install numpy pandas scikit-learn
conda install matplotlib
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `step1_load_split.py` | 加载数据、分层划分训练/验证集（80/20，seed=42） |
| `step2_preprocess_vectorize.py` | 原始/清洗两条流水线的 TF-IDF 向量化，保存中间产物 |
| `step3_train_compare.py` | 三个模型默认参数基线 + 清洗对照实验 |
| `step4_tuning.py` | 网格调参（3 模型 × 2 流水线 × 小网格 = 18 组配置） |
| `step5_evaluate_plot.py` | 最优配置深入评估 + 生成 3 张报告用图 |
| `step6_predict_submit.py` | 全量数据重训最优模型，生成提交文件 `predictions.csv` |
| `train_split.csv` / `val_split.csv` | 固定的数据划分结果（由 step1 生成） |
| `*.npz` / `*.pkl` | TF-IDF 稀疏矩阵与向量化器中间产物（由 step2 生成） |
| `tuning_results.csv` | 18 组调参结果总表（由 step4 生成） |
| `fig*.png` | 报告用图（由 step5 生成） |

## 运行步骤

在 `data-Project1` 目录下，按顺序执行：

```bash
# 1. 数据加载与划分（生成 train_split.csv / val_split.csv）
python step1_load_split.py

# 2. 特征工程（生成 .npz / .pkl 中间产物）
python step2_preprocess_vectorize.py

# 3. 基线模型 + 清洗对照
python step3_train_compare.py

# 4. 网格调参（约几分钟）
python step4_tuning.py

# 5. 评估与可视化（生成 3 张图）
python step5_evaluate_plot.py

# 6. 生成提交文件 predictions.csv
python step6_predict_submit.py
```

## 可复现性说明

- 所有模型固定 `random_state=42`
- 划分结果保存为文件，后续步骤共用同一份训练/验证集，保证公平比较
- 数据版本：`data-Project1`（助教提供）

## 最终结论

- 最优方案：**原始文本（不做清洗）+ 逻辑回归 C=10**
- 验证集准确率：**0.9315**，macro-F1：**0.9319**
- 提交文件：`predictions.csv`（无表头、单列、2457 行）
