# _*_ coding: utf-8 _*_
'''
时间:      2025/6/3 22:59
@author:  andinm
'''
import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(1e-5, 1-1e-5, 41)
# 当类别为2时
H_p = -p*np.log(p) - (1 - p) * np.log(1 - p)
plt.plot(p, H_p, color="black", label="H(p) = -p*np.log(p) - (1-p)*np.log(1-p)")
plt.legend(loc="upper left")
plt.xlabel("p"), plt.ylabel("H(p)")
plt.grid(True, alpha=1)
plt.show()

# 熵越大 不确定性越大









