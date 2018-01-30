import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy.optimize import minimize
import math


data = io.loadmat("hw1data.mat")
x = data['x']
y = data['y']

def fun1(varX):
    x1 = x.copy()
    y1 = y.copy()
    res = 0.0;
    for (i, j) in zip(x1,y1):
        res +=( j - (varX[0]*i + varX[1]))**2
    return res

#----------------------

x0 = [0.0, 0.0]

res = minimize(fun1, x0)
print(res.x)
plt.scatter(x, y)
resY = x*res.x[0] + res.x[1] 
plt.plot(x, resY)
plt.savefig("firstPlot.png")
plt.clf()
#----------------------
#x2 = x.copy()
#y2 = y.copy()

x0 = [0.0, 0.0]

def fun2(varX):
    x1 = x.copy()
    y1 = y.copy()
    res = 0.0;
    for (i, j) in zip(x1,y1):
        res += math.fabs(j - (varX[0]*i + varX[1]))
    return res

res = minimize(fun2, x0)
print(res.x)
plt.scatter(x, y)
resY = x*res.x[0] + res.x[1] 
plt.plot(x, resY)
plt.savefig("secondPlot.png")

# Final values pasted below: 
# 8.a) => [  3.48630142  61.16321456]
# 8.b) => [  3.59688303  17.97875709]

