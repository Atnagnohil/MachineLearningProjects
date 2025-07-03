# _*_ coding: utf-8 _*_
'''
时间:      2025/7/3 15:35
@author:  andinm
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
import random

# --- 关键修正 4: 配置Matplotlib以支持中文 ---
# 设置一个支持中文的字体，例如 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
# 这可以解决UserWarning和图表标题中文显示为方框的问题。
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


# 读取函数
def loadData(filename):
    dataMatIn = []; classLabels = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMatIn.append([float(lineArr[0]), float(lineArr[1])])
        classLabels.append(float(lineArr[2]))
    fr.close()
    return dataMatIn, classLabels
# 随机选择alpha的函数
def chooseJrand(i, m):
    # 从0到m中选择一个不等于i的
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j
# 剪辑函数
def clipAlpha(alphaJNewUnClip, L ,H):
    if alphaJNewUnClip > H:
        return H
    elif L <= alphaJNewUnClip <= H:
        return alphaJNewUnClip
    else:
        return L

# SMO算法函数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
        简版SMO算法
        Args:
            dataMatIn (list): 特征
            classLabels (list): 标签
            C (float): 惩罚参数
            toler (float): 容错率
            maxIter (int): 最大迭代次数
        Returns:
            b (float): 偏置项
            alphas (matrix): alpha向量
        """
    # 转换为矩阵
    dataMat = np.asmatrix(dataMatIn); labelMat = np.asmatrix(classLabels).reshape(-1, 1)
    m, n = dataMat.shape
    # 初始化alphas
    alphas = np.asmatrix(np.zeros(shape=(m, 1), dtype=float))
    # 初始化b为0
    b = 0
    # 初始化迭代次数
    iter = 0
    # --- 关键修正 1: 修改循环逻辑 ---
    # 原始代码的循环逻辑会在每次更新alpha后重置迭代计数器，
    # 如果算法无法快速收敛，将导致无限循环。
    # 新的逻辑是，进行固定次数的迭代，或者当一整轮迭代都没有alpha更新时提前结束。
    while iter < maxIter:
        alphaPairsChanged =0
        for i in range(m):
            # 计算gxi
            gxi = (np.multiply(alphas, labelMat).T @ (dataMat @ dataMat[i,:].T)).item() + b
            # 计算Ei
            Ei = gxi - (labelMat[i]).item()
            if (labelMat[i]*Ei < -toler and alphas[i] < C) or (labelMat[i]*Ei > toler and alphas[i] > 0):
                # 随机选择一个等待优化的alpha
                j = chooseJrand(i, m)
                gxj = (np.multiply(alphas, labelMat).T @ (dataMat @ dataMat[j, :].T)).item() + b
                Ej = gxj - (labelMat[j]).item()
                K11, K22, K12, K21 = (dataMat[i] @ dataMat[i].T, dataMat[j] @ dataMat[j].T,
                                      dataMat[i] @ dataMat[j].T, dataMat[j] @ dataMat[i].T)
                eta = K11 + K22 - 2*K12
                alphaIOld = alphas[i].copy()
                alphaJOld = alphas[j].copy()
                alphaJOldUncilp = alphaJOld + (labelMat[j] * (Ei - Ej) / eta)
                # 计算上下界
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphaJOld - alphaIOld)
                    H = min(C, C + alphaJOld - alphaIOld)
                else:
                    L = max(0, alphaJOld + alphaIOld - C)
                    H = min(C, alphaJOld + alphaIOld)
                alphaJNew = clipAlpha(alphaJOldUncilp, L, H)
                alphaINew = alphaIOld + labelMat[i]*labelMat[j]*(alphaJOld - alphaJNew)

                # --- 关键修正 2: 更新alphas矩阵 ---
                # 原始代码计算了新的alpha值，但没有将它们赋值回alphas矩阵，
                # 导致alphas永远是0，算法无法学习。
                alphas[i] = alphaINew
                alphas[j] = alphaJNew

                b1New = (-Ei - labelMat[i] * K11 * (alphaINew - alphaIOld) -
                         labelMat[j] * K21 * (alphaJNew - alphaJOld) + b)
                b2New = (-Ej - labelMat[i] * K12 * (alphaINew - alphaIOld) -
                         labelMat[j] * K22 * (alphaJNew - alphaJOld) + b)
                if alphaINew > 0 and alphaINew < C: b = b1New
                elif alphaJNew > 0 and alphaJNew < C: b = b2New
                else: b = (b1New + b2New) / 2.0
                alphaPairsChanged += 1
                # 在一轮完整的迭代后，打印进度
        if alphaPairsChanged == 0:
            print(f"在第 {iter + 1} 轮迭代后收敛。")
            break
        else:
            print(f"第 {iter + 1} 轮迭代完成, 更新了 {alphaPairsChanged} 对alpha。")
            iter += 1

    if iter == maxIter:
        print(f"达到最大迭代次数 {maxIter}。")

    return b, alphas


# 计算Omega
def calOmega(alphas,dataMatIn,classLabels):
    dataMat = np.asmatrix(dataMatIn); labelMat = np.asmatrix(classLabels).reshape(-1, 1)
    m, n = dataMat.shape
    # 初始化Omega都为1
    Omega = np.zeros((n, 1))
    # 循环计算
    for i in range(m):
        Omega += np.multiply(alphas[i] * labelMat[i], dataMat[i, :].T)
    return Omega

def plotPoint(dataMatIn, classLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    # --- 关键修正 3: 使用正确的标签为数据点着色 ---
    # 原始代码错误地使用了数据点本身(dataMatIn)来着色。
    ax.scatter(np.array(dataMatIn)[:, 0], np.array(dataMatIn)[:, 1], c=np.array(classLabels).squeeze(), cmap=cm_dark,
               s=30)
    plt.title("原始数据点")
    plt.show()

# 绘制决策平面
def plotDesignSurface(dataMat, classLabels , Omega, b):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(dataMat)[:, 0], np.array(dataMat)[:, 1], c=np.array(classLabels).squeeze(), cmap=cm_dark, s=30)

    x = np.arange(-2.0, 12.0, 0.1)
    # w0*x + w1*y + b = 0  => y = (-w0*x - b) / w1
    y = (-Omega[0] * x - b) / Omega[1]
    ax.plot(x, y.reshape(-1, 1))
    ax.axis([-2, 12, -8, 6])
    plt.show()
# 绘制支持向量
def plotSupportVector(dataMat, classLabels , Omega, b, alphas):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(dataMat)[:, 0], np.array(dataMat)[:, 1], c=np.array(classLabels).squeeze(), cmap=cm_dark, s=30)

    x = np.arange(-2.0, 12.0, 0.1)
    y = (-Omega[0] * x - b) / Omega[1]
    ax.plot(x, y.reshape(-1, 1))
    ax.axis([-2, 12, -8, 6])

    alphas_non_zeros_index = np.where(alphas > 0)
    for i in alphas_non_zeros_index[0]:
        circle = Circle((dataMat[i][0], dataMat[i][1]), 0.2, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,
                        alpha=0.5)
        ax.add_patch(circle)
    plt.show()

if __name__ == "__main__":
    dataMatIn, classLabels = loadData("Data/svm1.txt")
    b, alphas = smoSimple(dataMatIn, classLabels, 0.6, 0.001, 40)
    Omega = calOmega(alphas, dataMatIn, classLabels)
    print("\n--- 计算结果 ---")
    print("b:", b)
    # 打印非零的alpha值（支持向量）
    print("支持向量的Alphas:\n", alphas[alphas > 0])
    plotPoint(dataMatIn, classLabels)
    plotDesignSurface(dataMatIn, classLabels, Omega, b)
    plotSupportVector(dataMatIn, classLabels, Omega, b, alphas)









