# -*- coding: utf-8 -*-
# step4_tuning.py
# 实验一 · 第4步：三个模型 × 两条特征流水线 × 小参数网格 的调参对比
#
# 运行方式（在 data-Project1 文件夹下执行）：
#   & "C:\Users\HP\.conda\envs\project1\python.exe" step4_tuning.py
# 注意：这一步含多次模型训练，预计运行几分钟，请耐心等待输出。

import time
import pandas as pd
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ========== 1. 加载两条流水线的特征矩阵和标签 ==========
X_train_raw = sparse.load_npz('X_train_raw.npz')
X_val_raw = sparse.load_npz('X_val_raw.npz')
X_train_clean = sparse.load_npz('X_train_clean.npz')
X_val_clean = sparse.load_npz('X_val_clean.npz')

train_df = pd.read_csv('train_split.csv')
val_df = pd.read_csv('val_split.csv')
y_train = train_df['target'].values
y_val = val_df['target'].values

# 两条流水线：raw = 原始文本, clean = 清洗文本
pipelines = {
    'raw':   (X_train_raw,   X_val_raw),
    'clean': (X_train_clean, X_val_clean),
}

# ========== 2. 定义参数网格（数量级取值，见 TUNING.md）==========
# 朴素贝叶斯：alpha 是拉普拉斯平滑系数，越大对未见过词的容忍度越高
nb_grid = [0.1, 0.5, 1.0]
# SVM：C 是正则化参数，C 越大越"较真"地分对训练样本
svm_grid = [0.1, 1.0, 10.0]
# 逻辑回归：C 是正则化强度(的倒数)，C 越小正则越强、模型越简单
lr_grid = [0.1, 1.0, 10.0]

results = []

def run_and_record(pname, model_name, params, model):
    """训练 + 验证 + 记录（顺带计时，用于报告里的"性能 vs 复杂度"讨论）"""
    t0 = time.time()
    model.fit(Xtr, y_train)
    acc = accuracy_score(y_val, model.predict(Xva))
    cost = round(time.time() - t0, 1)
    results.append({'pipeline': pname, 'model': model_name,
                    'params': params, 'val_acc': acc, 'fit_time_s': cost})
    print('[%s] %-18s %-12s 验证准确率=%.4f  训练耗时=%.1fs'
          % (pname, model_name, params, acc, cost), flush=True)

# ========== 3. 网格搜索 ==========
for pname, (Xtr, Xva) in pipelines.items():
    print('\n===== 流水线: %s =====' % pname)

    for alpha in nb_grid:
        run_and_record(pname, 'MultinomialNB', 'alpha=%.1f' % alpha,
                       MultinomialNB(alpha=alpha))

    for C in svm_grid:
        run_and_record(pname, 'SVC(linear)', 'C=%.1f' % C,
                       SVC(kernel='linear', C=C, random_state=42))

    for C in lr_grid:
        run_and_record(pname, 'LogisticRegression', 'C=%.1f' % C,
                       LogisticRegression(C=C, max_iter=2000, random_state=42))

# ========== 4. 汇总与排序 ==========
df = pd.DataFrame(results)
df = df.sort_values('val_acc', ascending=False).reset_index(drop=True)
print('\n========== 调参结果总表（按验证准确率降序） ==========')
print(df.to_string(index=False))

df.to_csv('tuning_results.csv', index=False)

best = df.iloc[0]
print('\n========== 当前最优配置 ==========')
print('流水线: %s | 模型: %s | 参数: %s | 验证准确率: %.4f | 训练耗时: %.1fs'
      % (best['pipeline'], best['model'], best['params'], best['val_acc'], best['fit_time_s']))
print('结果已保存到 tuning_results.csv')
