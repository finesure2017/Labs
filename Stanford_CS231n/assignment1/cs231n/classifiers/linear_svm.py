import numpy as np
from random import shuffle

import sys

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
    # for every training and class, 
    # Loss equation will be: max(0, Sij - Sicorrectclassindex + 1)
    # max(0, scores[j] - correct_class_score + 1)
    # max(0, margin)
      if j == y[i]:
        # No need to compute loss if correct class
        # Also, since both are same => Sij - Sicorrectclassindex + 1 = 0 + 1
        # => Gradient of loss is 0
        # Therefore, no need to compute gradient as well
        continue

      # Compute loss
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        # Only include loss if more than margin => Loss is not 0
        loss += margin
        # Update for wrong class
        dW[:, j] += X[i]
        # Update for correct class
        dW[:, y[i]] -= X[i]
      else:
        # If margin < 0 => Max(0, < 0 ) = 0 => loss = 0 => gradient = 0
        # Thus, don't need to compute loss or gradient
        continue 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Normalize by number of training samples
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW += 2*reg*W

  #############################################################################
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  scores = np.dot(X, W) # (N, C)
  numTrain = X.shape[0]
  # Not sure if there's a better way to index, to create correctClassScore
  correctClassScore = scores[np.arange(numTrain), y] # (N, 1)
  margin = scores - np.reshape(correctClassScore, (-1, 1)) + 1 #(N, C)
  lossPerClass = np.maximum(0, margin) # (N, C)
  lossVector = np.sum(lossPerClass, axis=1) # (N, 1)
  totalLoss = np.average(lossVector) #(1)
  totalLoss -= 1.0 # Deduct 1, which is the delta as added correct class score into margin
  # Add regularization to the loss.
  totalLoss += reg * np.sum(W * W)
  loss = totalLoss
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  lossPerClass = np.maximum(0, margin) # (N, C)
  # Make 0 for any where margin <= 0
  # Make 1 for any where margin > 0
  # Make -1 for any where it is the correct class
  multiplication = np.zeros(lossPerClass.shape) # (N, C)
  yCorrectClass = np.zeros(lossPerClass.shape)

  posIndices = np.where(lossPerClass > 0)
  # Set whatever that needs to create an update to 1
  # Wrong instances: initialize everything X multiples to be a positve X for wrong instances
  multiplication[posIndices] = 1.0 
  # However, should not update those position that required update 
  # where the score computed was the correct class score. 
  # Set all the correct class positions to 0, since the gradient cancels out for those correct class location
  multiplication[np.arange(numTrain), y] = 0.0

  # Sum across the rows to get the number of wrong instances for each training example
  # (N, 1)
  # Correct Instances: Sum the number of times you need to multiply X for correct instances and negate to get -X
  values = (-1.0 * np.sum(multiplication, axis=1)) # (N, 1)
  # Set those rows to deduct x since it's correct examples
  # note: This line does not handle the score computed was correct class score
  # as you need it to all be 0.0 before calculating the values for below.
  multiplication[np.arange(numTrain), y] = values # Update the multiplication table for those correct instances
  dW = np.dot(np.transpose(X), multiplication) # Multiply with X based on the number of times you need to update
  # Normalize the gradient
  dW /= numTrain
  # Add regularization gradient
  dW += 2.0 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
