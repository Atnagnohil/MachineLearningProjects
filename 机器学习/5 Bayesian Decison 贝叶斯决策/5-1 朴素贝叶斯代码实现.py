# _*_ coding: utf-8 _*_
'''
时间:      2025/6/26 11:46
@author:  andinm
'''
import numpy as np
# 加载数据
def loadData():
    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'],
                  [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
                  [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
                  [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    return X, y

# 训练模型
def trainModel(trainset, trainlabels):
    '''

    :param trainset: 为传入的数据
    :param trainlabels: 对应的标签
    :return:
    '''
    n, m = trainset.shape
    # n个数据 m个特征
    # 首先计算先验概率
    prior_probability = {}
    labels = set(trainlabels)
    # print(labels)
    for label in labels:
        prior_probability[label] = len(trainlabels[trainlabels==label])
    # print(prior_probability)

    # 计算条件概率
    condition_probability_cnt = {}  # 存储的key: [类别，特征，特征值]
    for i in range(n):
        for j in range(m):
            if (trainlabels[i], j, trainset[i][j]) in condition_probability_cnt:
                condition_probability_cnt[(trainlabels[i], j, trainset[i][j])] += 1
            else:
                condition_probability_cnt[(trainlabels[i], j, trainset[i][j])] = 1
    condition_probability = {}
    for key in condition_probability_cnt.keys():
        condition_probability[key] = condition_probability_cnt[key] / prior_probability[key[0]]
    for label in labels:
        prior_probability[label] = prior_probability[label] / n
    return prior_probability, condition_probability, labels

X, y = loadData()
prior_probability, condition_probability, labels = trainModel(X, y)
# print(prior_probability)
# print(condition_probability)
# print(labels)

# 进行预测
def predict(data, prior_probability, condition_probability, labels):
    # data是传入的数据
    res = {}
    for label in labels:
        item = 1.0
        for idx, feature in enumerate(data):

            # print((label, idx, feature))
            item *= condition_probability[(label, idx, str(feature))]
        item *= prior_probability[label]
        res[label] = item
    print(res)
    return sorted(res.items(), key=lambda x: x[1], reverse=True)[0][0]
print(f"预测结果：{predict([2, "S"], prior_probability, condition_probability, labels)}")










