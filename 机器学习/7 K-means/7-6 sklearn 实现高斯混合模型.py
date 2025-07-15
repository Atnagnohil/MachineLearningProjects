# _*_ coding: utf-8 _*_
'''
时间:      2025/7/10 16:37
@author:  andinm
'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

# 生成数据
def generateData():
    # 生成 均值1.71 标准差为0.056的男生数据
    np.random.seed(0)   # 初始化随机种子
    mu_m = 1.71  # 期望
    sigma_m = 0.056  # 标准差
    num_m = 10000  # 数据个数为10000
    rand_data_m = np.random.normal(mu_m, sigma_m, num_m)  # 生成数据
    y_m = np.zeros(num_m)  # 生成标签

    # 生成均值为1.58 标准差为0.051的数据
    mu_w = 1.58  # 期望
    sigma_w = 0.051  # 标准差数据
    num_w = 10000  # 个数为10000
    rand_data_w = np.random.normal(mu_w, sigma_w, num_w)  # 生成数据
    y_w = np.ones(num_m)  # 生成标签

    # 混合数据
    data = np.append(rand_data_m, rand_data_w)
    data = data.reshape(-1, 1)
    y = np.append(y_m, y_w)
    return data, y

def train(data, K):
    g = GaussianMixture(n_components=K, covariance_type='full', tol=1e-6, max_iter=1000)
    g.fit(data)
    print(u'类别概率:\t', g.weights_[0])
    print(u'均值:\n', g.means_, '\n')
    print(u'方差:\n', g.covariances_, '\n')
    return g


def access(model, y, data):
    y_hat = model.predict(data)

    # 获取模型的两个簇的均值
    means = model.means_.flatten()

    # 根据均值大小重新映射标签
    if means[0] > means[1]:
        # 簇0对应男生(0)，簇1对应女生(1)
        y_hat_mapped = np.where(y_hat == 0, 0, 1)
    else:
        # 簇1对应男生(0)，簇0对应女生(1)
        y_hat_mapped = np.where(y_hat == 1, 0, 1)

    print("预测准确率:", accuracy_score(y, y_hat_mapped))
if __name__ == "__main__":
    data, y = generateData()
    # print(data)
    model = train(data, K)
    access(model, y, data)















