# _*_ coding: utf-8 _*_
'''
时间:      2025/7/10 16:05
@author:  andinm
'''
import numpy as np
from scipy.stats import multivariate_normal
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

def train(data, y):
    # 初始化
    iter = 1000
    n, d = data.shape
    # 初始化参数
    mu1 = data.min(axis=0)
    mu2 = data.max(axis=0)
    sigma1 = np.identity(d)
    sigma2 = np.identity(d)
    pi = 0.5

    for i in range(iter):
        # 计算gamma
        normal1 = multivariate_normal(mu1, sigma1)
        normal2 = multivariate_normal(mu2, sigma2)
        gamma1 = (pi * normal1.pdf(data)) / (pi * normal1.pdf(data) + (1 - pi) * normal2.pdf(data))
        # 计算mu
        mu1 = np.dot(gamma1, data) / np.sum(gamma1)
        mu2 = np.dot(1- gamma1, data) / np.sum(1 - gamma1)
        # 计算Σ
        sigma1 = np.dot(gamma1 * (data - mu1).T, data - mu1) / np.sum(gamma1)
        sigma2 = np.dot((1 - gamma1) * (data - mu2).T, data - mu2) / np.sum(1 - gamma1)
        # 计算pi
        pi = np.sum(gamma1) / n
        # if i == 0:
        #     print(gamma1)
        #     print(gamma1.shape)
        #     print(mu1)
        #     print(sigma1)
    print(u'类别概率:\t', pi)
    print(u'均值:\t', mu1, mu2)
    print(u'方差:\n', sigma1, '\n\n', sigma2, '\n')
    '''
    类别概率:	 0.49719420863322855
    均值:	 [1.58022223] [1.70858555]
    方差:
    [[0.00253127]] 

    [[0.00309491]] 
    '''
if __name__ == "__main__":
    data, y = generateData()
    # print(data)
    train(data, y)








