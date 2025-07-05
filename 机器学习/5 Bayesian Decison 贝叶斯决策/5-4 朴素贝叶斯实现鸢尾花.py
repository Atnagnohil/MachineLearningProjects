# _*_ coding: utf-8 _*_
'''
时间:      2025/6/26 16:31
@author:  andinm
'''

from sklearn.datasets import load_iris
from sklearn import naive_bayes as nb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
def loadData():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y
def dataSplit(X, y):
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return XTrain, XTest, yTrain, yTest

def trainModel(XTrain, yTrain):
    # 进行网格化搜索
    # 定义要搜索的参数网格
    param_grid = {
        'alpha': np.linspace(3, 4, 20),  # 平滑参数的范围
        'fit_prior': [True, False]  # 是否学习类先验概率
    }

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=nb.MultinomialNB(),  # 使用多项式朴素贝叶斯
        param_grid=param_grid,  # 参数网格
        cv=5,  # 5折交叉验证
        scoring='accuracy',  # 评估指标为准确率
        n_jobs=-1  # 使用所有可用的CPU核心
    )

    # 执行网格搜索
    grid_search.fit(XTrain, yTrain)

    # 输出最佳参数
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_  # 返回最佳模型

def Assess(XTest, yTest, model, label):
    y_hat = model.predict(XTest)
    # 打印预测报告
    print(classification_report(yTest, y_hat))

    # 绘制混淆矩阵
    mat = confusion_matrix(yTest, y_hat)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=label,
                yticklabels=label)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

    print(f"accuracy_score: {accuracy_score(y_hat, yTest)}")

if __name__ == "__main__":
    X, y = loadData()
    label = ["山鸢尾", "杂色鸢尾", "维吉尼亚鸢尾"]
    XTrain, XTest, yTrain, yTest = dataSplit(X, y)
    model = trainModel(XTrain, yTrain)
    Assess(XTest, yTest, model, label)












