# _*_ coding: utf-8 _*_
'''
时间:      2025/6/27 14:14
@author:  andinm
'''

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import naive_bayes as nb
from sklearn.metrics import accuracy_score
import numpy as np


def createVocabList(dateSet):
    # 创建空的的词汇列表
    vocaSet = set([])
    for document in dateSet:
        vocaSet = vocaSet | set(document)
    return sorted(list(vocaSet))


# 词集模型
def setOfWords2Vec(inputSet, vocaList):
    '''

    :param inputSet: 输入的词句
    :param vocaSet: 对应的词集
    :return: 对应的编码
    '''
    # 初始化向量
    returnVec = [0] * len(vocaList)
    for word in inputSet:
        if word in vocaList:
            returnVec[vocaList.index(word)] = 1
        else:
            print("词步不在词汇表中")
    return returnVec

# 词袋模型
def bagOfWords2Vec(inputSet, vocaList):
    returnVec = [0] * len(vocaList)
    for word in inputSet:
        if word in vocaList:
            returnVec[vocaList.index(word)] += 1
        else:
            print("词步不在词汇表中")
    return returnVec

# 对邮件内容的预处理
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def loadData():
    docList = []
    classList = []
    for i in range(1, 26):
        try:
            wordList = textParse(open('Data/email/spam/%d.txt' % i).read())
            docList.append(wordList)
            classList.append(1)

            wordList = textParse(open('Data/email/ham/%d.txt' % i).read())
            docList.append(wordList)
            classList.append(0)
        except FileNotFoundError as e:
            print(f"错误 文件未被找到 - {e}, 请确保 'Data' 文件夹在正确的路径下。")

    # 创建词集
    vocabList = createVocabList(docList)
    X = []
    for document in docList:
        # 使用词袋模型
        X.append(bagOfWords2Vec(document, vocabList))
    return X, classList, vocabList
def splitData(X, y):
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return XTrain, XTest, yTrain, yTest

def trainModel(Xtrain, yTrain):
    # 创建管道符
    param_grid = {
        "alpha": np.linspace(1e-10, 1e-2, 21), # 平滑参数的范围
        "fit_prior": [True, False]   # 是否使用先验概率
    }
    grid_search = GridSearchCV(
        estimator=nb.MultinomialNB(),
        param_grid=param_grid,
        cv=5, # 5折交叉验证
        scoring="accuracy", # 评估核心指标
        n_jobs=-1  # 使用所有可用的cpu核心
    )
    grid_search.fit(Xtrain, yTrain)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def Assess(model, XTest, yTest):
    y_hat = model.predict(XTest)
    return accuracy_score(y_hat, yTest)

if __name__ == "__main__":
    X, classList, vocabList = loadData()
    # 分割数据集
    XTrain, XTest, yTrain, yTest = splitData(X, classList)
    bestModel, bestParam, bestScore = trainModel(XTrain, yTrain)
    scoreOfTest = Assess(bestModel, XTest, yTest)
    print(f"最佳参数: {bestParam}")
    print(f"最佳交叉验证正确率: {bestScore:.4%}")
    print(f"在测试集模型的正确率: {scoreOfTest: .4%}")




