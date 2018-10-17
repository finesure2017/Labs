# Homework 3 Question 1 c)
# Note: I included code to show train loss does decrease.

# note: I also included the code for sketching Question 1 a) 

import numpy as np
import matplotlib.pyplot as plt

# Code to perform full batch mode gradient descent on the model. 
# Assume training dataset is given as design matrix X, target vector y. 
# Initialize W, b to all zeros. 
# Code should be vecorized. 
# In other words, no for loops over training examples or input dimensions.
# Hint: Use np.where() if it is helpful

def HuberLossGradientDescent(X, y, W, b, delta = 1.0):
    '''
    Performs Full Batch Gradient Descent on Huber Loss
    X = (N, d) design matrix, N is number of training points, d is dimension of each training point's domain
    y = (N), each y_{i} is a scalar value
    delta => the tolerance for norm of a to switch from L2 loss to L1 loss
    Since Huber Loss has piecewise gradients, 
    np.where() is used to get to each gradient

    Returns:
    dW is gradients for W, size is (d, 1)
    db is gradients for bias, size is (b)
    '''
    N = X.shape[0] # Number of data points
    X = np.concatenate((X, np.ones((N, 1))),axis=1) # add constant one to last dimension to represent bias in weights
    W = np.append(W, b) # concatenates W and b
    d = X.shape[1] # Input dimension including bias dimension
    dW = np.zeros(d) 

    # Implement piecewise gradient
    a = np.dot(X, W)
    a = np.reshape(a, (a.shape[0], 1))
    y = np.reshape(y, a.shape)

    # Get norm of a
    normA = np.sqrt(np.power(a, 2.0))
    # Get the three cases indices
    caseAIndex = np.where((normA <= delta))[0]
    caseBIndex = np.where((normA > delta) & (a >= 0.0))[0]
    caseCIndex = np.where((normA > delta) & (a < 0.0))[0]

    # Add to gradient
    if caseAIndex.shape[0] > 0:
        dW = np.reshape(dW, (dW.shape[0], 1)) + np.dot(np.transpose(X[caseAIndex]), (a[caseAIndex] - y[caseAIndex]))
    if caseBIndex.shape[0] > 0:
        dW = np.reshape(dW, (dW.shape[0], 1)) + np.reshape(np.dot(np.transpose(X[caseBIndex]), (delta * np.ones(X[caseBIndex].shape[0]))), (dW.shape[0], 1))
    if caseCIndex.shape[0] > 0:
        dW = np.reshape(dW, (dW.shape[0], 1)) + np.reshape(np.dot(np.transpose(X[caseCIndex]), ((-1.0 * delta) * np.ones(X[caseCIndex].shape[0]))), (dW.shape[0], 1))

    # Average gradient out
    dW = dW/float(X.shape[0])

    db = dW[-1] # Extract gradient for bias
    dW = dW[:-1] # Extract gradient for weights
    return dW, db

def huberLoss(yTruth, yPred, delta):
    """
    Returns huber loss
    """
    losses = np.where(np.abs(yPred - yTruth) < delta , 0.5*((yPred - yTruth)**2), delta*(np.abs(yPred - yTruth) - 0.5*(delta)))
    meanhuberLoss = np.mean(losses)
    return meanhuberLoss

def meanSquareLoss(yTruth, yPred):
    meanSquareLoss = 0.5 * ((yPred - yTruth)**2)
    return meanSquareLoss

def sketchHuberAndSquareLoss():
    '''
    This code is for sketching Huber and Square Loss
    '''
    yPred = np.arange(-2, 2, 0.01) # make errors
    y = np.repeat(0, yPred.shape[0])  # Repeat same value many times
    lossMse = [meanSquareLoss(y[i], yPred[i]) for i in range(len(yPred))]
    lossHuber = [huberLoss(y[i], yPred[i], 1.0) for i in range(len(yPred))]
    plt.plot(yPred-y, lossMse, label="MseLoss")
    plt.plot(yPred-y, lossHuber, label="HuberLoss, delta=1")
    plt.title("meanSquareLoss vs huberLoss with delta=1")
    plt.ylabel("Loss")
    plt.xlabel("yPred - yTruth")
    plt.legend()
    plt.savefig("huberSquareLoss.png")

if __name__ == "__main__":
    sketchHuberAndSquareLoss() # For question1 a)
    np.random.seed(0) # Replicability
    N = 10 # num points
    b = 1.0
    d = 2 # Make it two dimension for now
    X = np.reshape(np.arange(N*d), (N, d)) # Make it single dimensional for now
    w = np.ones((d, 1))
    y = np.dot(X, w) + b
    numIteration = 16 # Number of iteration to perform gradient descent
    learnRate = 1e-2
    # Initialize weights to 0
    # Note: Bias is the last dimension in weights
    # we can do this as the lost equation can be collapse into the same function using X = 1 for the bias gradients
    W = np.zeros(d)
    b = 0.0
    for i in range(numIteration):
        dW, db = HuberLossGradientDescent(X, y, W, b)
        learnRateDw = np.multiply(learnRate, dW)
        W = np.reshape(W, (W.shape[0], 1)) - learnRateDw
        b = b - np.multiply(learnRate, db)
        yPred = np.dot(X, W) + b
        trainLoss = np.linalg.norm(y-yPred)
        print("Iteration: {}, TrainLoss: {}".format(i, trainLoss))
