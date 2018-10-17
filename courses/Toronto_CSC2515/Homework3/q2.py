# Homework Qesution 2 b), 2 c)
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_boston

def l2(A,B):
    '''
    Compute the squared of L2 norm between 2 matrices
    Input: A is a Nxd matrix
           B is a Mxd matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def LRLSBatch(xTest, xTrain, yTrain, tau, lam=1e-5):
    '''
    Locally Weighted Least Square Regression
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter (lambda on paper) 
    output is y_hat the prediction on test_datum
    '''
    numTest = xTest.shape[0]
    yPred = np.zeros(numTest)
    for currIndex in range(numTest):
        yPred[currIndex] = LRLS(xTest[currIndex], xTrain, yTrain, tau, lam=1e-5)
    return yPred

def calculateTrainWeights(test_datum, xTrain, tau, lam):
    # Compute the weighting for each training point
    a = np.zeros(xTrain.shape[0]) # (N x 1) 

    # Calculate distance between the single test point wrt to every training point
    squaredDistance = l2(np.reshape(test_datum, (1, xTrain.shape[1])), xTrain)

    # Ai is a (numTrain) shaped matrix that is the distance between the single test point 
    # with respect to all training points
    Ai = np.reshape((-1.0/float(2.0 * np.power(tau,2.0))) * squaredDistance, (xTrain.shape[0]))

    # Set the weighting parameter using logsumexp for numerical stability
    denom =  scipy.misc.logsumexp(Ai)
    a = Ai - denom # note: Ai already accounts for log on numerator, Ai = log(exp(Ai))
    # Take the exponent since we used the log earlier for numerical stability
    a = np.exp(a)

    # Diagonalize a to perform matrix operations
    A = np.diagflat(a)
    return A

def LRLS(test_datum, xTrain, yTrain, tau, lam=1e-5):
    '''
    Locally Weighted Least Square Regression
    Input: test_datum is a dx1 test vector
           xTrain is the N_train x d design matrix
           yTrain is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter (lambda on paper) 
    output is y_hat the prediction on test_datum
    '''
    # A is a diagonal matrix, where the diagonal entries are the weights
    # of each training point  with respect to a single test point
    # A is size (n, n), where n is the number of training points
    A = calculateTrainWeights(test_datum, xTrain, tau, lam)

    # Compute the closed form parameter values without inverting
    lhs = np.dot(np.dot(np.transpose(xTrain), A), xTrain) + (lam * np.identity(xTrain.shape[1]))
    rhs = np.dot(np.dot(np.transpose(xTrain), A), yTrain)
    W = np.linalg.solve(lhs,rhs) # (d, 1)

    # Compute Optimal prediction for the single test datum
    testDatumPred =  np.dot(test_datum, W)
    return testDatumPred 

def lossCalculate(yTruth, yPred):
    '''
    Calculate mean loss based on true and prediction for locally weighted linear regression
    Do not include regularization in loss calculation according to piazza: 
    Link: https://piazza.com/class/jlp72odwmqo2v2?cid=361
    '''
    difference = yTruth - yPred
    meanLoss = np.mean(np.multiply(0.5, np.power(difference, 2.0)))
    return meanLoss

def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    Output: 
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    N = x.shape[0] # Number of data points
    # Add 1 extra dimension to every training point
    x = np.concatenate((np.ones((N, 1)),x),axis=1) # add constant one feature to represent bias in weights
    d = x.shape[1] # Input dimension including bias dimension

    # One loss for each tau value
    numTau = taus.shape[0]
    trainLosses = np.zeros(numTau)
    testLosses = np.zeros(numTau)
    
    # Randomly shuffle the N datasets
    idx = np.random.permutation(range(N))
    numTest = int(round(val_frac * N))
    numTrain = N - numTest
    # Shuffle
    x = x[idx]
    y = y[idx]

    # Hold out validation set (only 1 validation set)
    # Split train and validation (test) set after shuffling
    xTrain = x[:numTrain]
    yTrain = y[:numTrain]
    xTest = x[numTrain:]
    yTest = y[numTrain:]

    for currTauIndex in range(numTau):
        tau = taus[currTauIndex]
        # In this exercise, we fixed lambda (hard coded to 1e-5) and only set tau value. 
        # Can change lambda if you want
        # Get predictions on train and test set
        yTrainPred = LRLSBatch(xTrain, xTrain, yTrain, tau, lam=1e-5)
        yTestPred = LRLSBatch(xTest, xTrain, yTrain, tau, lam=1e-5)

        # Calculate train and test loss based on prediction
        trainLosses[currTauIndex] = lossCalculate(yTrain, yTrainPred)
        testLosses[currTauIndex] = lossCalculate(yTest , yTestPred) 
    return trainLosses, testLosses

if __name__ == "__main__":
    np.random.seed(0) # Replicability
    # Load boston housing prices dataset
    boston = load_boston()
    y = boston['target'] # Output values
    x = boston['data']

    taus = np.logspace(1.0, 3, 200) # Base 10, 10^1 = 10, 10^3=1000, 200 points 
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)

    # Plot the train and test losses
    # A loss for each tau value
    plt.semilogx(taus, train_losses, label="Train Loss")
    plt.semilogx(taus, test_losses, label="Validation Loss")
    plt.title("Semilog of tau vs loss")
    plt.xlabel("Tau")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('trainValidationLoss.png')
