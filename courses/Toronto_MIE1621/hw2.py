# Code for newton method for hw 2 question 3
import math
import numpy as np

numIteration = 7
initialX1 = 1.0/3.0
initialX2 = 1.0/3.0
currX1 = initialX1
currX2 = initialX2
f = - np.log(1.0-currX1-currX2) - np.log(currX1) - np.log(currX2)
print("Initial Points: " +"(x1 = " + str(currX1) + ", x2 = " + str(currX2) + ")")

count = 0
while(True):
    print("Iteration: " + str(count))
    print("(x1 = " + str(currX1) + ", x2 = " + str(currX2) + ")")
    print("f = " + str(f))
    dfdx1 = (1.0/(1-currX1-currX2)) - (1.0/currX1)
    dfdx2 = (1.0/(1-currX1-currX2)) - (1.0/currX2)
    gradF = np.array([[dfdx1], [dfdx2]])
    gradF = np.reshape(gradF, (2, 1))
    t = (1.0)/np.square(1 - currX1 - currX2)
    hessianF = np.array([[t + (1.0/(currX1 * currX1)), t], [t, t + (1.0/(currX2*currX2))]])
    hessianF = np.reshape(hessianF, (2, 2))
    k = np.dot(np.linalg.inv(hessianF),gradF)
    nextX1 = currX1 - k[0]
    nextX2 = currX2 - k[1]
    f = - np.log(1.0-currX1-currX2) - np.log(currX1) - np.log(currX2)
    currX1 = nextX1
    currX2 = nextX2
    count += 1
    if count > numIteration:
        break
