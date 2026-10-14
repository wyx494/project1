# -*- coding: utf-8 -*-
# step3_train_compare.py
# 实验一 · 第3步：三个经典分类模型的训练与初步对比
#
# 运行方式（在 data-Project1 文件夹下执行）：
#   & "C:\Users\HP\.conda\envs\project1\python.exe" step3_train_compare.py

import pandas as pd
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ========== 1. 加载步骤2保存的特征矩阵和标签 ==========
# 直接用保存好的 .npz 稀疏矩阵，避免重复向量化，也保证与步骤2完全一致
X_train_clean = sparse.load_npz('X_train_clean.npz')
X_val_clean = sparse.load_npz('X_val_clean.npz')
X_train_raw = sparse.load_npz('X_train_raw.npz')
X_val_raw = sparse.load_npz('X_val_raw.npz')

train_df = pd.read_csv('train_split.csv')
val_df = pd.read_csv('val_split.csv')
y_train = train_df['target'].values
y_val = val_df['target'].values

# ========== 2. Part A：三个模型，默认参数，清洗特征 ==========
# 先用合理默认值把流程跑通（实验推荐流程的第1步：建立基线）
models = {
    'MultinomialNB': MultinomialNB(),
    'SVC(linear)': SVC(kernel='linear', random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
}

print('========== Part A: 三个模型默认参数在验证集上的准确率（清洗特征） ==========')
results = []
for name, model in models.items():
    model.fit(X_train_clean, y_train)                 # 只在训练集上拟合
    acc = accuracy_score(y_val, model.predict(X_val_clean))   # 在验证集上评估
    results.append((name, acc))
    print('%-22s 验证集准确率: %.4f' % (name, acc))

# ========== 3. Part B：清洗 vs 不清洗 对照实验 ==========
# 控制变量：模型和参数完全一样，只有"是否清洗"不同
# 用逻辑回归做对照（三个模型里它对特征变化最敏感、最有代表性）
print()
print('========== Part B: 清洗 vs 不清洗（逻辑回归，其余条件一致） ==========')
lr_raw = LogisticRegression(max_iter=1000, random_state=42)
lr_raw.fit(X_train_raw, y_train)
acc_raw = accuracy_score(y_val, lr_raw.predict(X_val_raw))
print('原始文本 + LR: 验证集准确率 = %.4f' % acc_raw)

lr_clean = LogisticRegression(max_iter=1000, random_state=42)
lr_clean.fit(X_train_clean, y_train)
acc_clean = accuracy_score(y_val, lr_clean.predict(X_val_clean))
print('清洗文本 + LR: 验证集准确率 = %.4f' % acc_clean)
print('清洗带来的提升: %+.4f' % (acc_clean - acc_raw))

# 提示：模型默认参数跑完后的结论先记下来，步骤4会做正式的调参对比
