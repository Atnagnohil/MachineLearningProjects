# _*_ coding: utf-8 _*_
'''
时间:      2025/5/26 21:05
@author:  andinm
'''
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




def loadData():
    data = np.loadtxt(f"Data/data1.txt",delimiter=',')
    n = data.shape[1] - 1
    X = data[:, 0:n]
    y = data[:, -1]
    return X, y
X, y = loadData()

# 创建scaler并应用到数据
scaler = StandardScaler(with_mean=True, with_std=True)  # 标准化（均值0，方差1）
X_std = scaler.fit_transform(X)

# 定义评估函数（确保输入为一维数组）
def calMSE(y_true, y_predict):
    return np.mean((y_true - y_predict) ** 2)

def calRMSE(y_true, y_predict):
    return np.sqrt(calMSE(y_true, y_predict))

def calMAE(y_true, y_predict):
    return np.mean(np.abs(y_true - y_predict))

# 最小二乘法
def linear():
    # 加载模型
    model1 = linear_model.LinearRegression()
    '''LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    '''
    # 训练模型
    model1.fit(X, y)
    print(model1.coef_)  # 斜率
    print(model1.intercept_)  # 截距
    plt.scatter(X, y)
    y_hat = model1.predict(X)
    plt.plot(X, y_hat)
    plt.show()

    print(f"MSE = {calMSE(y, y_hat)}")
    print(f"RMSE = {calRMSE(y, y_hat)}")
    print(f"MAE = {calMAE(y, y_hat)}")

# Ridge回归
def ridge():
    model2 = linear_model.Ridge(alpha=0.001)  # alpha是正则化强度注意：这里的alpha表示正则化强度normalize设置为True表示对训练数据进行标准化
    model2.fit(X_std, y)
    print(model2.coef_)
    print(model2.intercept_)
    plt.scatter(X_std, y)
    y_hat = model2.predict(X_std)
    plt.plot(X_std, y_hat)
    plt.show()
    print(f"MSE = {calMSE(y, y_hat)}")
    print(f"RMSE = {calRMSE(y, y_hat)}")
    print(f"MAE = {calMAE(y, y_hat)}")
# ridge()

# Lasso回归
def lasso():
    model3 = linear_model.Lasso(alpha=0.2)  # 正则化强度
    model3.fit(X_std, y)

    # 生成连续数据开始绘图
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_std_plot = scaler.transform(X_plot)  # 使用创建的scaler将生成的线性数据 标准化
    y_plot = model3.predict(X_std_plot)  # 使用标准化后的线性数据预测数据
    # 绘制图像
    plt.scatter(X_std, y, label="Data")
    plt.plot(X_std_plot, y_plot, label="Lasso", color="black")
    plt.xlabel("x(Std)")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # 指标评估
    y_hat = model3.predict(X_std)
    print(f"MSE = {calMSE(y, y_hat)}")
    print(f"RMSE = {calRMSE(y, y_hat)}")
    print(f"MAE = {calMAE(y, y_hat)}")
    # 输出标准化空间的参数
    print("\n标准化空间参数：")
    print("Coefficients:", model3.coef_)
    print("Intercept:", model3.intercept_)

    # 转换到原始数据空间（可选）
    coef_original = model3.coef_ / scaler.scale_
    intercept_original = model3.intercept_ - (coef_original * scaler.mean_).sum()
    print("\n原始数据空间参数：")
    print("Coefficients:", coef_original)
    print("Intercept:", intercept_original)

lasso()

# 弹性网
def elactisnet():
    model4 = linear_model.ElasticNet(alpha=0.2)  # 正则化强度
    model4.fit(X_std, y)

    # 生成连续数据开始绘图
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_std_plot = scaler.transform(X_plot)  # 使用创建的scaler将生成的线性数据 标准化
    y_plot = model4.predict(X_std_plot)  # 使用标准化后的线性数据预测数据
    # 绘制图像
    '''这一步有点多余'''
    plt.scatter(X_std, y, label="Data")
    plt.plot(X_std_plot, y_plot, label="Lasso", color="black")
    plt.xlabel("x(Std)")
    plt.ylabel("y")
    plt.legend()
    plt.show()


    # 指标评估
    y_hat = model4.predict(X_std)
    print(f"MSE = {calMSE(y, y_hat)}")
    print(f"RMSE = {calRMSE(y, y_hat)}")
    print(f"MAE = {calMAE(y, y_hat)}")
    # 输出标准化空间的参数
    print("\n标准化空间参数：")
    print("Coefficients:", model4.coef_)
    print("Intercept:", model4.intercept_)

    # 转换到原始数据空间（可选）
    coef_original = model4.coef_ / scaler.scale_
    intercept_original = model4.intercept_ - (coef_original * scaler.mean_).sum()
    print("\n原始数据空间参数：")
    print("Coefficients:", coef_original)
    print("Intercept:", intercept_original)
# elactisnet()

