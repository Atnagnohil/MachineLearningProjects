# _*_ coding: utf-8 _*_
'''
时间:      2025/7/10 16:48
@author:  andinm
'''
'''
UserWarning: KMeans is known to have a memory leak on Windows with MKL, 
when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
警告原因：
数据量较小（20行）
MKL库在数据量小于可用线程数时会出现内存泄漏
默认情况下，sklearn会使用所有可用的CPU线程

在kmeans中移除了n_jobs参数
只能通过设置全局变量控制线程
'''
# 设置使用2个线程（可调整数字）
# === 关键修复：设置所有线程控制环境变量 且需要在其他库导入之前 ===
'''
原因：
    SciPy/NumPy/scikit-learn 这些科学计算库在导入时会加载底层的数学库（如 OpenMP、MKL 等）
    这些数学库在初始化时会确定并行线程数
    如果环境变量在这之前未设置，它们会使用默认值（通常是所有可用核心）
    
    当您import pandas时，它可能已经间接导入了NumPy
    NumPy导入时会初始化底层数学库
    此时环境变量尚未设置，所以使用默认线程数
    后续设置环境变量只是设置了系统变量，但无法改变已初始化的数学库配置
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"   # 必须设置此变量
os.environ["MKL_NUM_THREADS"] = "1"   # Intel MKL线程数

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def loadData():
    df = pd.read_csv("Data/football_team_data.csv",index_col="国家")
    return df
def featureNormalize(df):
    X = df.values
    X = StandardScaler().fit_transform(X)
    return X
def trainModel(X, K):
    model = KMeans(n_clusters=K, max_iter=100)       # 考虑将队伍聚为3类
    model.fit(X)
    return model
def display(model, df):
    df["聚类"]=model.labels_
    print(df)

if __name__ == "__main__":
    df = loadData()
    # print(df)
    X = featureNormalize(df)
    model = trainModel(X, 2)
    display(model, df)


