import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import math
import cvxpy as cp

def partA(x, y):
    a = cp.Variable()
    b = cp.Variable()
    c = cp.Variable()
    yHat = a*(x**2) + b*(x) + c
    objective = cp.Minimize(cp.sum((y - yHat)**2))
    problem = cp.Problem(objective, None)
    result = problem.solve()
    return (a.value, b.value, c.value)

def partB(x, y):
    a = cp.Variable()
    b = cp.Variable()
    c = cp.Variable()
    yHat = a*(x**2) + b*(x) + c
    objective = cp.Minimize(cp.sum(y - yHat))
    # The noise is positive, so the difference in prediction should be more thna 0
    constraint = [y - yHat >= 0]
    problem = cp.Problem(objective, constraint)
    result = problem.solve()
    return (a.value, b.value, c.value)

data = io.loadmat("hw1data.mat")
x = np.squeeze(data['x'])
y = np.squeeze(data['y'])
indices = np.argsort(x)
x = x[indices]
y = y[indices]

(a, b, c) = partA(x.copy(), y.copy())
yhatA= a*(x**2) + b*x + c

(a, b, c) = partB(x.copy(), y.copy())
yhatB= a*(x**2) + b*x + c

# Plot results
plt.plot(x, y, label='y')
plt.plot(x, yhatA, label='yhatA')
plt.plot(x, yhatB, label='yhatB')
plt.legend()
plt.savefig("allInOne.png")
