import numpy as np
from random import shuffle
import sys

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  numTrain = X.shape[0]
  numInputDim = X.shape[1]
  numClass = W.shape[1]
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for currTrainIndex in range(numTrain):
      den = 0.0
      for currClassIndex in range(numClass):
          den += np.exp(np.dot(X[currTrainIndex], W[:, currClassIndex]))
      inNum = np.dot(X[currTrainIndex], W[:, y[currTrainIndex]])
      # num = np.exp(inNum)
      # loss += -(1.0) * np.log(num/den)
      loss += np.log(den) - inNum

      for currClassIndex in range(numClass):
          gradientNum = X[currTrainIndex] * np.exp(np.dot(X[currTrainIndex], W[:,  currClassIndex]))
          dW[:, currClassIndex] += (gradientNum/den)
          if currClassIndex == y[currTrainIndex]:
              dW[:, currClassIndex] -= X[currTrainIndex]

  loss /= numTrain
  dW /= numTrain 

  loss += reg * np.sum(W * W)
  dW += 2.0 * reg *  W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  numTrain = X.shape[0]
  numClass = W.shape[1]
  
  loss += np.sum(logSumExp(np.dot(X, W), givenAxis = 1))
  loss -= np.sum(np.dot(X, W)[np.arange(numTrain), y])

  num = np.exp(np.dot(X, W)) # (N, C)
  den = np.sum(num, axis=1) # (N, 1)
  result = num/np.reshape(den, (-1, 1)) # (N,C) 
  # X => (N, D)
  dW += np.dot(np.transpose(X), result) # (D, C)

  '''
  # Note: Attempted approach BELOW IS FOR GRADIENT COMPUTATION WRONG because, 
  # when you take the gradient, you  must take with respect to each index of (C, D)
  # Hence, this means that you do not need to maintain the D
  # The D comes from the Xi that is out of thte gradient equation, not from the logsumexp
  # For gradient, need maintain the dimension D, whereas for loss, you summed over it

  # Every training instance has a unique denominator
  # Wrong approach below, should be (N, 1)
  den = np.sum(np.exp(np.multiply(np.reshape(X, (-1, 1)), np.reshape(W, (1, -1)))), axis = 2) # (N, D)
  # Wrong approach below, should be (N, C)
  # Also, this reshaping and broadcasting uses too much space. Just use np.dot(np.transpose(X), W)
  num = np.exp(np.multiply(np.reshape(X, (-1, 1)), np.reshape(W, (1, -1)))) # (N, D, C)
  probability = num/den # (N, D, C)
  # Sum over all training instances
  dW += np.sum(probability, axis = 0)
  # '''

  # Account for cases where the classes are the correct class.
  # The gradient has an extra deduction of X[i] 
  mask = np.zeros((numTrain, numClass)) # (N, C)
  mask[np.arange(numTrain), y] = 1.0 # Put 1 at each N at the relevant locations
  dW -= np.dot(np.transpose(X), mask) # (D, C) # Deduct only at those locations

  loss /= numTrain
  dW /= numTrain 

  loss += reg * np.sum(W * W)
  dW += 2.0 * reg *  W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

def logSumExp(x, givenAxis = 0):
    # This works mainly because softmax function is invariant to constants
    # softmax(x) = softmax(x + c), where c is a constant
    # log(e^x) = log(e^(x-c+c)) = log(e^(x-c)*e^(c)) = log(e^(c)) + log(e^(x-c)) = c + log(e^(x-c))
    # Hence, you make most of the x values closer to 0 by deducting the max
    # This prevents exp(largeX) => Infinity
    maxValue = np.max(x)
    return maxValue + np.log(np.sum(np.exp(x-maxValue), axis = givenAxis))
