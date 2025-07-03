from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# def loadData():
#     data = np.loadtxt(f"Data/data1.txt", delimiter=',')
#     n = data.shape[1] - 1
#     X = data[:, 0:n]
#     y = data[:, -1] # 不 reshape 为列向量
#     return X, y
# X, y = loadData()
#
# # 数据标准化（仅需对 X 标准化）
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
#
# # 定义评估函数（确保输入为一维数组）
# def calMSE(y_true, y_predict):
#     return np.mean((y_true - y_predict) ** 2)
#


# def calRMSE(y_true, y_predict):
#     return np.sqrt(calMSE(y_true, y_predict))
#
# def calMAE(y_true, y_predict):
#     return np.mean(np.abs(y_true - y_predict))
#
# # Lasso 回归
# def lasso():
#     model3 = linear_model.Lasso(alpha=0.2)
#     model3.fit(X_std, y)
#
#     # 预测及评估
#     y_hat = model3.predict(X_std)
#     print(f"MSE = {calMSE(y, y_hat):.4f}")
#     print(f"RMSE = {calRMSE(y, y_hat):.4f}")
#     print(f"MAE = {calMAE(y, y_hat):.4f}")
#
#     # 打印参数
#     print("\n标准化空间参数：")
#     print("Coefficients:", model3.coef_)
#     print("Intercept:", model3.intercept_)
#
#     # 转换到原始数据空间
#     coef_original = model3.coef_ / scaler.scale_
#     intercept_original = model3.intercept_ - (coef_original @ scaler.mean_)
#     print("\n原始数据空间参数：")
#     print("Coefficients:", coef_original)
#     print("Intercept:", intercept_original)
#
# lasso()

print(np.linspace(14,14.8,21))