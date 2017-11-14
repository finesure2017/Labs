# Code for newton method for hw 3 question 1
import math
import numpy as np

initialX1 = 1.0
currX1 = initialX1
f = math.pow(currX1, 4) - 32.0 * math.pow(currX1, 2)
print("Initial Points: " +"(x1 = " + str(currX1) + ")")
numIteration = 7
count = 0
while(True):
    print("Iteration: " + str(count))
    print("(x1 = " + str(currX1) + ")")
    print("f = " + str(f))
    dfdx1 = 4.0*math.pow(currX1, 3) - 64.0*currX1
    df2dx1x1 = 12.0*math.pow(currX1, 2) - 64.0
    print("dfdx1 = " + str(dfdx1))
    print("df2dx1x1 = " + str(df2dx1x1))
    #nextX1 = currX1 - (f / dfdx1)
    nextX1 = currX1 - (dfdx1 / df2dx1x1)
    currX1 = nextX1
    f = math.pow(currX1, 4) - 32.0 * math.pow(currX1, 2)
    count += 1
    if count > numIteration:
        break
