# Code for newton method for hw 3 question 2
import math
import numpy as np

numIteration = 9
initialX1 = 1.0/3.0
initialX2 = 1.0/3.0
currX1 = initialX1
currX2 = initialX2
f = math.pow((currX1 - 2.0), 4) + (math.pow((currX1 - 2.0), 2) * math.pow(currX2, 2)) + math.pow((currX2 + 1.0), 2)
print("Initial Points: " +"(x1 = " + str(currX1) + ", x2 = " + str(currX2) + ")")
count = 0
while(True):
    print("Iteration: " + str(count))
    print("(x1 = " + str(currX1) + ", x2 = " + str(currX2) + ")")
    print("f = " + str(f))
    dfdx1 = (4.0 * math.pow((currX1 - 2.0), 3)) + (2.0 * (currX1 - 2.0) * math.pow(currX2, 2))
    dfdx2 = (2.0 * math.pow((currX1 - 2.0), 2) * currX2) + (2.0 * (currX2 + 1.0))
    gradF = np.array([[dfdx1], [dfdx2]])
    gradF = np.reshape(gradF, (2, 1))
    df2dx1x1 = (12.0 * math.pow((currX1 - 2.0), 2)) + (2.0 * math.pow(currX2, 2))
    df2dx1x2 = (4.0 * (currX1 - 2.0) * currX2)
    df2dx2x2 = (2.0 * math.pow((currX1 - 2.0), 2)) + 2.0
    hessianF = np.array([[df2dx1x1, df2dx1x2], [df2dx1x2, df2dx2x2]])
    hessianF = np.reshape(hessianF, (2, 2))
    print("Grad:")
    print(gradF)
    print("Hessian:")
    print(hessianF)
    k = np.dot(np.linalg.inv(hessianF),gradF)
    nextX1 = currX1 - k[0]
    nextX2 = currX2 - k[1]
    currX1 = nextX1
    currX2 = nextX2
    f = math.pow((currX1 - 2.0), 4) + (math.pow((currX1 - 2.0), 2) * math.pow(currX2, 2)) + math.pow((currX2 + 1.0), 2)
    count += 1
    if count > numIteration:
        break
