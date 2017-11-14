# Code for newton method for hw 2 question 3
import math
import numpy as np

def namestr(obj): 
    namespace = globals()
    return [name for name in namespace if namespace[name] is obj]

def pprint(obj):
    print(namestr(obj), obj.shape)
    print(obj)

#----------------------------------------------------------------------------------------------------------------
# Problem 1 c) 
#----------------------------------------------------------------------------------------------------------------
T = np.array([[2.0 + math.pow(10, -3), 3.0, 4.0], [3.0, 5.0 + math.pow(10, -3), 7.0],
    [4.0, 7.0, 10.0 + math.pow(10, -3)]])
t = np.array([[20.0019], [34.0004], [48.0202]])
L = np.eye(3)
theta = 1.0

pprint(T)
pprint(t)
pprint(L)

normT = np.dot(np.transpose(T), T)
normL = np.dot(np.transpose(L), L)
beforeInv = normT + theta*(normL)
afterInv = np.linalg.inv(beforeInv)
Tt = np.dot(np.transpose(T), t)
x = np.dot(afterInv, Tt)

pprint(normT)
pprint(normL)
pprint(beforeInv)
pprint(afterInv)
pprint(Tt)
pprint(x)

#----------------------------------------------------------------------------------------------------------------
# Problem 1 d)
#----------------------------------------------------------------------------------------------------------------
invNormT = np.linalg.inv(normT)
x = np.dot(invNormT, Tt)
pprint(invNormT)
pprint(x)


#----------------------------------------------------------------------------------------------------------------
# Problem 5 a) Quasi Newton Methods
#----------------------------------------------------------------------------------------------------------------

Q = np.array([[5, -3], [-3, 2]])
b = np.array([[0], [1]])
x = np.array([[0], [0]])
H = np.eye(2)
f = 0.5 * np.dot(np.dot(np.transpose(x), Q), x) - np.dot(np.transpose(b), x) + np.log(np.pi)

pprint(Q)
pprint(b)
pprint(x)
pprint(H)
pprint(f)

print("Iteration 0:")
pprint(x)
# Start iterating
for i in range(2): # Iterate twice only for Quasi Newton to converge
    xold = x
    gradFk = np.dot(Q, xold) - b
    Hinv = np.linalg.inv(H)

    dk = - np.dot(Hinv, gradFk) # TODO: Not sure if includes negative
    #----------------------------------------------------------
    # Calculate stepSize for exact step 
    # Problem 5a
    stepSize = (1.0/(np.dot(np.dot(np.transpose(dk), Q), dk))) * (np.dot(
        np.transpose(dk), b) - np.dot(np.dot(np.transpose(dk), Q), xold))
    #----------------------------------------------------------
    pprint(stepSize)
    xnew = xold - stepSize * np.dot(Hinv, gradFk)
    gradFk1 = np.dot(Q, xnew) - b
    y = gradFk1 - gradFk
    s = xnew - xold
    # BFGS Update
    Hnext = H + (1.0/(np.dot(np.transpose(y), s))) * np.dot(y, np.transpose(y)) - (1.0/(np.dot(np.dot(np.transpose(s), H), s))) * np.dot(np.dot(np.dot(H, s), np.transpose(s)), H)

    pprint(xold)
    pprint(H)
    pprint(Hinv)
    pprint(gradFk)
    pprint(dk)
    pprint(xnew)
    pprint(gradFk1)
    pprint(y)
    pprint(s)

    H = Hnext
    x = xnew
    print("Iteration: ", i+1) # Iteration starts from 1
    pprint(xnew)

#----------------------------------------------------------------------------------------------------------------
# Problem 5 b) Newton Method
#----------------------------------------------------------------------------------------------------------------
Q = np.array([[5, -3], [-3, 2]])
b = np.array([[0], [1]])
x = np.array([[0], [0]])
H = Q # Initialize as Q
f = 0.5 * np.dot(np.dot(np.transpose(x), Q), x) - np.dot(np.transpose(b), x) + np.log(np.pi)

pprint(Q)
pprint(b)
pprint(x)
pprint(H)
pprint(f)

print("Iteration 0:")
pprint(x)
# Start iterating
for i in range(2): # Iterate once only Newton to converge
    xold = x
    gradFk = np.dot(Q, xold) - b
    Hinv = np.linalg.inv(H)

    dk = - np.dot(Hinv, gradFk) # TODO: Not sure if includes negative
    #----------------------------------------------------------
    # Problem 5b
    # Uncomment below for newton's method
    stepSize = np.array([1.0]) # Old newton method for Problem 5 b)
    #----------------------------------------------------------
    pprint(stepSize)
    xnew = xold - stepSize * np.dot(Hinv, gradFk)
    # Problem 5b

    pprint(xold)
    pprint(H)
    pprint(Hinv)
    pprint(gradFk)
    pprint(dk)
    pprint(xnew)
    x = xnew
    print("Iteration: ", i+1) # Iteration starts from 1
    pprint(xnew)
#----------------------------------------------------------------------------------------------------------------
