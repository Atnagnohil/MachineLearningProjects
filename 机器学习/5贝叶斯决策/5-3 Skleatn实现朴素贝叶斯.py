# _*_ coding: utf-8 _*_
'''
时间:      2025/6/26 16:09
@author:  andinm
'''

import numpy as np
from sklearn import naive_bayes as nb
from sklearn.preprocessing import LabelEncoder

def loaddata():
    X = np.array([[1,'S'],[1,'M'],[1,'M'],[1,'S'],
         [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
         [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
         [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    return X, y
X, y = loaddata()

# 对字符数据进行编码
X[:, 1] = LabelEncoder().fit_transform(X[:, 1])
X = X.astype(int)
# print(X)

# 训练模型
model = nb.MultinomialNB()
model.fit(X, y)

# 将[2, "s"]转换为[2, 2]
print(*model.predict([[2, 2]]))












