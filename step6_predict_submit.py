# -*- coding: utf-8 -*-
# step6_predict_submit.py
# 实验一 · 第6步：用全量训练数据重训最优模型，预测测试集，生成提交文件
#
# 运行方式（在 data-Project1 文件夹下执行）：
#   & "C:\Users\HP\.conda\envs\project1\python.exe" step6_predict_submit.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ========== 1. 加载全量训练数据 + 测试数据 ==========
# 注意：这里读的是最初的 train_data.csv（7368条 = 训练5894 + 验证1474）
# 调参已完成、方案已定稿，验证集没有用途了，把它"还回去"让模型见更多数据
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data_unlabeled.csv')

X_full = train_df['text'].astype(str).tolist()
y_full = train_df['target'].values
X_test = test_df['text'].astype(str).tolist()

# ========== 2. 特征工程（最优流水线：raw）==========
# 向量化只在训练数据上 fit；测试数据只 transform —— 测试集信息绝不进入特征统计
vectorizer = TfidfVectorizer(max_features=20000, min_df=2)
X_full_tfidf = vectorizer.fit_transform(X_full)
X_test_tfidf = vectorizer.transform(X_test)
print('训练矩阵:', X_full_tfidf.shape, ' 测试矩阵:', X_test_tfidf.shape)

# ========== 3. 训练最优模型（调参结论：LR + C=10）==========
model = LogisticRegression(C=10, max_iter=2000, random_state=42)
model.fit(X_full_tfidf, y_full)
print('模型训练完成')

# ========== 4. 预测并做合理性检查 ==========
preds = model.predict(X_test_tfidf)

# 合理性检查1：行数必须等于测试集样本数 2457
print('\n预测数量:', len(preds), '（应为 2457）')

# 合理性检查2：类别分布应与训练集比例大致接近
print('预测类别分布:')
print(pd.Series(preds).value_counts().sort_index())
print('训练集类别比例（参考）:')
print(pd.Series(y_full).value_counts(normalize=True).sort_index().mul(100).round(1))

# ========== 5. 保存提交文件 ==========
# 格式要求：无表头、单列、0~9 整数 —— 与助教示例 predictions.csv 一致
pd.DataFrame(preds).to_csv('predictions.csv', index=False, header=False)
print('\n已生成提交文件 predictions.csv（无表头，单列，%d 行）' % len(preds))
