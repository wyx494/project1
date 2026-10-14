# -*- coding: utf-8 -*-
# step1_load_split.py
# 实验一 · 第1步：数据加载、探索与训练/验证集划分
#
# 运行方式：在 data-Project1 文件夹下执行
#   python step1_load_split.py
# 说明：使用相对路径读取 csv，请确保当前工作目录在数据所在文件夹。

import pandas as pd
from sklearn.model_selection import train_test_split

# ========== 1. 加载数据 ==========
# train_data.csv 是带标签的训练数据，列：text, target
# test_data_unlabeled.csv 是待预测的无标签测试数据，列：text
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data_unlabeled.csv')

# 把文本和标签拆开
X = train_df['text'].astype(str)   # 特征：文本
y = train_df['target']             # 标签：0~9 的类别编号

# ========== 2. 数据探索 ==========
print('训练集规模:', train_df.shape)
print('测试集规模:', test_df.shape)
print()
print('各类别样本数:')
print(y.value_counts().sort_index())   # 按类别编号排序，方便看是否均衡
print()

# ========== 3. 划分训练集 / 验证集 ==========
# 为什么用分层抽样(stratify=y)？
#   如果随机划分，验证集中某类可能偏多/偏少，导致验证结果不可信；
#   分层抽样保证验证集里每个类别的比例与全量数据一致，结果更可靠。
# 为什么固定 random_state=42？
#   随机划分每次结果不同；固定种子后，任何人重跑都能得到完全相同的划分，保证实验可复现。
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,       # 20% 作为验证集，80% 作为训练集
    random_state=42,     # 固定随机种子
    stratify=y           # 按类别比例分层抽样
)

print('划分后 训练集样本数:', len(X_train))
print('划分后 验证集样本数:', len(X_val))
print()
print('验证集类别分布（应与整体比例一致）:')
print(y_val.value_counts().sort_index())
print()

# ========== 4. 保存划分结果 ==========
# 保存下来，后续所有步骤都读取这份固定的划分，
# 保证"同一条验证数据"被所有模型、所有参数共用，比较才公平。
pd.DataFrame({'text': X_train, 'target': y_train}).to_csv('train_split.csv', index=False)
pd.DataFrame({'text': X_val, 'target': y_val}).to_csv('val_split.csv', index=False)
print('已保存 train_split.csv 和 val_split.csv')
