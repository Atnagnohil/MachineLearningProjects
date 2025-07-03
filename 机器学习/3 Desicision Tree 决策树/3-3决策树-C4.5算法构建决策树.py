# _*_ coding: utf-8 _*_
'''
时间:      2025/6/15 22:28
@author:  andinm
'''
import math

# 加载数据
def loadData():
    dataSet = [[0, 0,0,0, 'no'],
               [0, 0,0,1,'no'],
               [0, 1,0,1, 'yes'],
               [0, 1,1,0, 'yes'],
               [0, 0,0,0, 'no'],
               [1, 0,0,0, 'no'],
               [1, 0,0,1, 'no'],
               [1, 1,1,1, 'yes'],
               [1, 0,1,2, 'yes'],
               [1, 0,1,2, 'yes'],
               [2, 0,1,2, 'yes'],
               [2, 0,1,1, 'yes'],
               [2, 1,0,1, 'yes'],
               [2, 1,0,2, 'yes'],
               [2, 0,0,0,'no']]
    feature_name = ['age','job','house','credit']
    return dataSet, feature_name

# 计算数据的熵  # 加上axis 方便计算某一特征下的熵
def entropy(dataSet, axis=-1):
    # 数据集条数
    m = len(dataSet)
    labelCnts = {}
    for featVec in dataSet:
        curLabel = featVec[axis]
        if curLabel not in labelCnts:
            labelCnts[curLabel] = 1
        else:
            labelCnts[curLabel] += 1
    e = 0.0
    for key in labelCnts.keys():
        prob = float(labelCnts[key]) / m
        e -= prob * math.log2(prob)
    return e
# dataSet, feature_name = loadData()
# print(entropy(dataSet))


# 划分数据集
def splitDataSet(dataset, axis, value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
# print(splitDataSet(dataSet, 2, 0))


# 选择最好的特征对应的下标(即为信息增益算法)
def chooseBestFeature(dataSet):
    # 首先计算经验熵
    baseEntropy = entropy(dataSet)
    # 计算条件经验熵
    bestInfoGain = 0.0  # 最大的信息增益比
    bestFeature = -1    # 最大的信息增益时对应的特征
    n = len(dataSet[0]) - 1  # n为所有的特征
    # 遍历每个特征
    for i in range(n):
        # 获取当前特征的所有值
        featList = [example[i] for example in dataSet]
        # 当前特征的可能取值
        uniqueVal = set(featList)
        # 定义一临时变量保存当前的条件熵
        newEntropy = 0.0
        # 循环每一个可能的取值
        for value in uniqueVal:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算条件熵（2行代码）
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * entropy(subDataSet)
        # 计算信息增益比
        Hi = entropy(dataSet, axis=i)

        # 修复方案：添加安全检查
        if abs(Hi) < 1e-5:  # 特征熵接近0
            continue  # 跳过该特征

        infoGain = (baseEntropy - newEntropy) / Hi
        # 保存当前最大的信息增益及对应的特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # 返回最优特征
    return bestFeature
# print(chooseBestFeature(dataSet))


# 投票表决
def classVote(classList):
    classCnt = {}
    for vote in classList:
        if vote not in classCnt:
            classCnt[vote] = 0
        classCnt[vote] += 1
    # print(classCnt)
    # 排序
    '''sortedClassCount = sorted(classCnt.items(), key=lambda x: x[1], reverse=True)    # 字典经过sorted方法之后会返回一个二维的列表
    return sortedClassCount[0][0]'''

    '''return max(classCnt, key=classCnt.get)'''

    return max(classCnt, key=lambda x: classCnt[x])  # 因为字典默认迭代的就是键值，所以逻辑上就相当于max(classCnt.keys(), key=lambda x: classCnt[x])


# classList = np.array(['yes', 'no', 'yes', 'no', 'yes'])
# print(classVote(classList))


# 构建决策树
def trainTree(dataSet, feature_name):
    # 取出数据集对应的标签
    classList = [example[-1] for example in dataSet]
    # 所有类别都一致
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 特征数只有一个
    if len(dataSet[0]) == 2:
        # 投票表决
        return classVote(classList)
    bestFeat = chooseBestFeature(dataSet)
    bestFeatName = feature_name[bestFeat]
    myTree = {bestFeatName:{}}
    featVals = [example[bestFeat] for example in dataSet]
    uninqueVals = set(featVals)
    for val in uninqueVals:
        sub_feature_name = feature_name[:]
        myTree[bestFeatName][val] = trainTree(splitDataSet(dataSet, bestFeat, val), sub_feature_name)
    return myTree
myDat,feature_name = loadData()
myTree = trainTree(myDat,feature_name)
print(myTree)
# {'house': {0: {'job': {0: 'no', 1: 'yes'}}, 1: 'yes'}}



# 进行预测
def predict(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = predict(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel
print(predict(myTree,feature_name,[1,1,0,1]))











