# _*_ coding: utf-8 _*_
'''
时间:      2025/7/16 21:43
@author:  andinm
'''

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
def loadData():
    data = np.loadtxt("Data/wine.data", delimiter=",")
    return data[:, 1:], data[:, 0]
# 分割数据集
def splitData(X, y):
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    return XTrain, XTest, yTrain, yTest
# 训练模型
def trainModel(X, y):
    '''使用AdaBoost训练数据集'''
    # 定义基分类器
    baseModel = DecisionTreeClassifier()
    model = AdaBoostClassifier(estimator=baseModel, n_estimators=50, algorithm="deprecated", learning_rate=0.5)
    model.fit(X, y)
    return model
def predict(X, model):
    '''<UNK>AdaBoost<UNK>'''
    yPred = model.predict(X)
    return yPred
if __name__ == '__main__':
    X, y = loadData()
    XTrain, XTest, yTrain, yTest = splitData(X, y)
    model = trainModel(XTrain, yTrain)
    yPred = predict(XTest, model)
    print(f"在测试集的精准度{accuracy_score(yTest, yPred):0.4%}")
    '''在测试集的精准度94.4444%
'''