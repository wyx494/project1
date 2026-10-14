# -*- coding: utf-8 -*-
# step5_evaluate_plot.py
# 实验一 · 第5步：最优模型深入评估 + 可视化（报告用图）
#
# 运行方式（在 data-Project1 文件夹下执行）：
#   & "C:\Users\HP\.conda\envs\project1\python.exe" step5_evaluate_plot.py
#
# 本步做三件事：
#   1) 对三个模型各自的最优配置做更全面的评估（准确率 + macro-F1）
#   2) 对最优模型（raw + LR C=10）输出分类报告、混淆矩阵、错误分析
#   3) 生成三张报告用图：模型对比柱状图 / 混淆矩阵热力图 / C调参曲线

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

import matplotlib
matplotlib.use('Agg')          # 终端环境下用非交互后端，只保存图片不弹窗
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# ========== 1. 加载数据（只用最优流水线 raw）==========
X_train = sparse.load_npz('X_train_raw.npz')
X_val = sparse.load_npz('X_val_raw.npz')
train_df = pd.read_csv('train_split.csv')
val_df = pd.read_csv('val_split.csv')
y_train = train_df['target'].values
y_val = val_df['target'].values

# ========== 2. 三个模型的最优配置（来自步骤4调参结论）==========
best_configs = {
    'LogisticRegression\nC=10': LogisticRegression(C=10, max_iter=2000, random_state=42),
    'SVC linear\nC=10': SVC(kernel='linear', C=10, random_state=42),
    'MultinomialNB\nalpha=0.1': MultinomialNB(alpha=0.1),
}

names = list(best_configs.keys())
accs, f1s = [], []
preds = {}
for name, model in best_configs.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    preds[name] = y_pred
    accs.append(accuracy_score(y_val, y_pred))
    f1s.append(f1_score(y_val, y_pred, average='macro'))
    print('%-28s 准确率=%.4f  macro-F1=%.4f' % (name.replace('\n', ' '), accs[-1], f1s[-1]))

# 选出全局最优 = 验证准确率最高的模型
best_name = names[int(np.argmax(accs))]
print('\n最优模型:', best_name.replace('\n', ' '))

# ========== 3. 最优模型的分类报告与错误分析 ==========
y_pred_best = preds[best_name]
print('\n===== 分类报告（最优模型，验证集） =====')
print(classification_report(y_val, y_pred_best, digits=4))

cm = confusion_matrix(y_val, y_pred_best)
# 找出最容易混淆的类别对（非对角线中最大者）
np.fill_diagonal(cm, -1)
worst_pair = np.unravel_index(np.argmax(cm), cm.shape)
np.fill_diagonal(cm, 0)
print('最容易混淆的类别对: 真类 %d 被错分为 %d，共 %d 个样本'
      % (worst_pair[0], worst_pair[1], cm[worst_pair]))

# ========== 4. 画图 ==========
# --- 图1: 三个模型最优配置对比（准确率 + macro-F1） ---
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, accs, width, label='Accuracy', color='#4C72B0')
ax.bar(x + width/2, f1s, width, label='Macro-F1', color='#55A868')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.85, 0.95)
ax.set_ylabel('Score')
ax.set_title('Best Config Comparison on Validation Set (raw pipeline)')
for i in range(len(names)):
    ax.text(i - width/2, accs[i] + 0.002, '%.4f' % accs[i], ha='center', fontsize=8)
    ax.text(i + width/2, f1s[i] + 0.002, '%.4f' % f1s[i], ha='center', fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig('fig1_model_comparison.png', dpi=150)
plt.close(fig)

# --- 图2: 混淆矩阵热力图（最优模型） ---
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(confusion_matrix(y_val, y_pred_best), cmap='Blues')
ax.set_xticks(np.arange(10)); ax.set_yticks(np.arange(10))
ax.set_xticklabels(np.arange(10)); ax.set_yticklabels(np.arange(10))
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix - %s (val set)' % best_name.replace('\n', ' '))
for i in range(10):
    for j in range(10):
        v = confusion_matrix(y_val, y_pred_best)[i, j]
        ax.text(j, i, str(v), ha='center', va='center',
                fontsize=8, color='white' if v > 60 else 'black')
fig.colorbar(im)
fig.tight_layout()
fig.savefig('fig2_confusion_matrix.png', dpi=150)
plt.close(fig)

# --- 图3: 逻辑回归 C 调参曲线（raw vs clean） ---
tuning = pd.read_csv('tuning_results.csv')
lr = tuning[tuning['model'] == 'LogisticRegression']
fig, ax = plt.subplots(figsize=(7, 4.5))
for pipe, color, marker in [('raw', '#4C72B0', 'o'), ('clean', '#C44E52', 's')]:
    sub = lr[lr['pipeline'] == pipe].sort_values('params')
    Cs = [float(p.split('=')[1]) for p in sub['params']]
    ax.plot(Cs, sub['val_acc'], color=color, marker=marker, label='pipeline=' + pipe)
ax.set_xscale('log')
ax.set_xticks([0.1, 1.0, 10.0])
ax.set_xticklabels(['0.1', '1', '10'])
ax.set_xlabel('C (regularization strength, log scale)')
ax.set_ylabel('Validation Accuracy')
ax.set_title('LogisticRegression: accuracy vs C')
ax.set_ylim(0.85, 0.94)
ax.legend()
fig.tight_layout()
fig.savefig('fig3_lr_c_curve.png', dpi=150)
plt.close(fig)

print('\n已生成图表: fig1_model_comparison.png / fig2_confusion_matrix.png / fig3_lr_c_curve.png')
print('（这三张图可直接用于实验报告）')
