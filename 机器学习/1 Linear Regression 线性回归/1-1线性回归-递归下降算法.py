# _*_ coding: utf-8 _*_
'''
时间:      2025/5/20 18:57
@author:  andinm
'''


# 梯度下降算法求解最小值  j = θ**2 + θ*2 +5的最小值
'''
设置初始θ, 设置步长a, 设置迭代次数m，，求J的导数
'''
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,8,40)
y = x**2 + x*2 + 5
x1 = 3
a = 0.3
m = 100
points = [x1]
def f(x):
    return x**2 + x*2 + 5
def df(x):
    return 2*x + 2
while m > 0:
    x1 = x1 - a*df(x1)
    points.append(x1)
    m -= 1
print(x1)

f_points = [f(x) for x in points]
print(f_points)
plt.plot(x,y)
plt.scatter(np.array(points), np.array(f_points),color="red")
plt.grid(True,alpha=0.7)
plt.show()










