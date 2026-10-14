# -*- coding: utf-8 -*-
# step2_preprocess_vectorize.py
# 实验一 · 第2步：文本预处理 + TF-IDF 特征工程
#
# 运行方式（在 data-Project1 文件夹下执行）：
#   & "C:\Users\HP\.conda\envs\project1\python.exe" step2_preprocess_vectorize.py

import re
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
import pandas as pd

# ========== 1. 加载上一步划分好的数据 ==========
train_df = pd.read_csv('train_split.csv')
val_df = pd.read_csv('val_split.csv')
X_train_text = train_df['text'].astype(str).tolist()
X_val_text = val_df['text'].astype(str).tolist()

# ========== 2. 文本清洗函数 ==========
def clean_text(text):
    # (1) 去掉邮件头：新闻组文本开头是一段元信息（From/Subject/Organization...），
    #     它们不是新闻内容本身，留着只会当噪声。以第一个空行为界，空行之前丢弃。
    parts = text.split('\n\n', 1)
    if len(parts) > 1:
        text = parts[1]
    # (2) 转小写（TfidfVectorizer 默认也会做，这里显式写出便于理解）
    text = text.lower()
    # (3) 去掉所有非字母字符（标点、数字、符号），只保留字母和空白
    text = re.sub(r'[^a-z\s]', ' ', text)
    # (4) 去掉停用词（the/a/an/and... 这类高频但无区分度的词）
    #     sklearn 内置英文停用词表 ENGLISH_STOP_WORDS，无需额外安装
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

# ========== 3. 两条特征工程流水线（控制变量）==========
# 流水线A：不做清洗，原始文本直接向量化（基线）
# 流水线B：先清洗，再向量化（对照）
# 两条线的向量化参数完全一致，只有"是否清洗"这一个变量 —— 控制变量原则
vectorizer_raw = TfidfVectorizer(max_features=20000, min_df=2)
vectorizer_clean = TfidfVectorizer(max_features=20000, min_df=2)

# ========== 4. 拟合 + 变换（关键原则！）==========
# fit_transform 只发生在训练集上：学词汇表、学每个词的 IDF 权重
# 验证集只调用 transform：用训练集学好的规则转换，绝不重新拟合
# 否则验证集的信息提前进入特征统计，属于数据泄露，会虚高验证分数
print('--- 流水线A：原始文本 + TF-IDF ---')
X_train_raw = vectorizer_raw.fit_transform(X_train_text)
X_val_raw = vectorizer_raw.transform(X_val_text)
print('词汇表大小:', len(vectorizer_raw.vocabulary_))

print('--- 流水线B：清洗文本 + TF-IDF ---')
X_train_clean = vectorizer_clean.fit_transform([clean_text(t) for t in X_train_text])
X_val_clean = vectorizer_clean.transform([clean_text(t) for t in X_val_text])
print('词汇表大小:', len(vectorizer_clean.vocabulary_))

# ========== 5. 查看特征效果 ==========
def sparsity(mat):
    """非零元素占比越低，矩阵越稀疏"""
    return 1.0 - mat.nnz / (mat.shape[0] * mat.shape[1])

print('\n矩阵规模: 训练', X_train_raw.shape, ' 验证', X_val_raw.shape)
print('训练集稀疏度: raw=%.6f  clean=%.6f' % (sparsity(X_train_raw), sparsity(X_train_clean)))
print()
print('清洗后第一个样本的开头 100 个词（对比原始文本的邮件头+正文）:')
print(' '.join(clean_text(X_train_text[0]).split()[:100]))
print()

# ========== 6. 保存中间产物（供步骤3复用）==========
sparse.save_npz('X_train_raw.npz', X_train_raw)
sparse.save_npz('X_val_raw.npz', X_val_raw)
sparse.save_npz('X_train_clean.npz', X_train_clean)
sparse.save_npz('X_val_clean.npz', X_val_clean)
with open('vectorizer_raw.pkl', 'wb') as f:
    pickle.dump(vectorizer_raw, f)
with open('vectorizer_clean.pkl', 'wb') as f:
    pickle.dump(vectorizer_clean, f)
print('已保存: X_train_raw.npz / X_val_raw.npz / X_train_clean.npz / X_val_clean.npz')
print('已保存: vectorizer_raw.pkl / vectorizer_clean.pkl')
