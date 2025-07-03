# _*_ coding: utf-8 _*_
'''
时间:      2025/7/3 22:57
@author:  andinm
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
import random

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def loadData(fileName):
    dataMatIn = []; classLabels = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMatIn.append([float(lineArr[0]), float(lineArr[1])])
        classLabels.append(float(lineArr[2]))
    fr.close()
    return dataMatIn,classLabels

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

#核函数
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.asmatrix(np.zeros((m, 1)))
    if kTup[0] == 'lin':#线性核
        K = X * A.T
    elif kTup[0] == 'rbf':#高斯核
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-2 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class OptStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.dataMat = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.asmatrix(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.asmatrix(np.zeros((self.m, 2))) #第一列是标志位，0无效 1有效
        self.K =  np.asmatrix(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.dataMat, self.dataMat[i, :], kTup)

# 修改 calcEk 函数
def calcEk(oS, k):
    # 获取标量值
    fXk = np.squeeze(np.multiply(oS.alphas, oS.labelMat).T @ oS.K[:, k] + oS.b).item()
    Ek = fXk - np.squeeze(oS.labelMat[k]).item()
    return Ek
            # ----------------- 优化最大处：选择一个和Ei差值最大的Ej
# 修改 chooseJ 函数
def chooseJ(i, oS, Ei):
    maxj, Ej, maxDeltaE = -1, 0, 0
    oS.eCache[i, 0] = 1
    # 使用 .item() 确保提取标量值
    oS.eCache[i, 1] = Ei.item() if hasattr(Ei, 'item') else Ei
    validEcacheList = np.nonzero(oS.eCache[:, 0])[0]
    if (len(validEcacheList)) > 1:
        for j in validEcacheList:
            if j == i: continue
            Ek = calcEk(oS, j)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxj = j
                maxDeltaE = deltaE
                Ej = Ek
        return maxj, Ej
    else:
        j = chooseJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 修改 updateEk 函数
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k, 0] = 1
    # 使用 .item() 确保提取标量值
    oS.eCache[k, 1] = Ek.item() if hasattr(Ek, 'item') else Ek


# 修改 innerL 函数 - 使用预计算的核矩阵
def innerL(oS, i):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = chooseJ(i, oS, Ei)
        alphaIOld, alphaJOld = oS.alphas[i].copy().item(), oS.alphas[j].copy().item()

        # 使用预计算的核矩阵
        K11 = oS.K[i, i]
        K22 = oS.K[j, j]
        K12 = oS.K[i, j]
        eta = K11 + K22 - 2 * K12

        if eta <= 0:
            return 0

        alphaJOldUncilp = alphaJOld + (oS.labelMat[j] * (Ei - Ej) / eta)

        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, alphaJOld - alphaIOld)
            H = min(oS.C, oS.C + alphaJOld - alphaIOld)
        else:
            L = max(0, alphaJOld + alphaIOld - oS.C)
            H = min(oS.C, alphaJOld + alphaIOld)
        if L == H:
            return 0

        alphaJNew = clipAlpha(alphaJOldUncilp, L, H)
        alphaINew = alphaIOld + oS.labelMat[i] * oS.labelMat[j] * (alphaJOld - alphaJNew)
        oS.alphas[i] = alphaINew
        oS.alphas[j] = alphaJNew
        updateEk(oS, i)
        updateEk(oS, j)

        if (abs(oS.alphas[j] - alphaJOld) < 0.00001):
            return 0

        # 使用预计算的核值更新b
        b1New = (-Ei - oS.labelMat[i] * K11 * (alphaINew - alphaIOld) -
                 oS.labelMat[j] * K12 * (alphaJNew - alphaJOld) + oS.b)
        b2New = (-Ej - oS.labelMat[i] * K12 * (alphaINew - alphaIOld) -
                 oS.labelMat[j] * K22 * (alphaJNew - alphaJOld) + oS.b)

        if alphaINew > 0 and alphaINew < oS.C:
            oS.b = b1New
        elif alphaJNew > 0 and alphaJNew < oS.C:
            oS.b = b2New
        else:
            oS.b = (b1New + b2New) / 2.0
        return 1
    else:
        return 0


#完整的改进后的使用核函数的SMO算法
def smoPro(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = OptStruct(np.asmatrix(dataMatIn), np.asmatrix(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  #遍历全部数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(oS, i)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  #遍历非边界数据集
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(oS, i)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  #进行切换
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

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


if __name__ == "__main__":
    dataMatIn, classLabels = loadData("Data/svm2.txt")
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    ax.scatter(np.array(dataMatIn)[:, 0], np.array(dataMatIn)[:, 1], c=np.array(classLabels).squeeze(), cmap=cm_dark, s=30)
    # plt.show()
    ###########
    b, alphas = smoPro(dataMatIn, classLabels, 40, 0.0001, 10000, ('rbf', 1.3))
    w = calOmega(alphas, dataMatIn, classLabels)
    print('b=', b)
    print('alphas=', alphas)
    print('w=', w)

    ###########
    # 画支持向量
    alphas_non_zeros_index = np.where(alphas > 0)
    for i in alphas_non_zeros_index[0]:
        circle = Circle((dataMatIn[i][0], dataMatIn[i][1]), 0.03, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,
                        alpha=0.5)
        ax.add_patch(circle)
    plt.show()
