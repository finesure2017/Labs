from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """ Initialize the model. Weights are initialized to small random values and biases are initialized to zero.
    Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    firstHiddenLayer = np.dot(X, W1) + b1

    # ReLU
    firstHiddenLayer[np.where(firstHiddenLayer < 0)] = 0.0
    finalLayer = np.dot(firstHiddenLayer, W2) + b2
    scores = finalLayer

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    #############################################################################
    # Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # Softmax Loss Classifer 
    # Note: Softmax Loss Classifier != L2 Loss != Cross Entropy Loss
    # ECE521, you did L2 Loss by creating a 1 hot vector of the correct labels
    # ECE521, also did Cross Entropy Loss and showed it was better for that problem.
    # BUT ECE521 did not cover softmax loss classifer which you are doing here. 
    # Softmax Loss Classifer simply maximizes the probability (M.L.E.) of the correct classes.

    # Get prediction of each class for each dataset
    predictions = mySoftmax(scores)
    # Iterate through dataset from 0 to N-1, for each dataset, index out the probability for the correct class.
    correctPredictions = predictions[np.arange(N), y]
    numClass = np.max(y) + 1
    # The closer the probability of the correct class is to 1.0, the lower the loss
    # Notice that softmax loss only cares bout the value of the correct class, and ignores
    # the probability given to all the other classes.
    loss += -1.0 * np.sum(np.log(correctPredictions))
    # Normalize by number of training samples
    loss /= N
    # Add the L2 regularization loss
    loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    '''
    # L2 Loss would have been
    numClasses = np.max(y) + 1
    oneHotVector = np.zeros((N, numClasses))
    oneHotVector[np.arange(N), y] = 1.0
    # Notice: L2Loss cares bout both the correct class (1.0 - probabilityOfCorrectClass)**2
    #         as well as the wrong classes (0.0 - probabilityOfWrongClass)**2
    L2Loss = (oneHotVector - predictions)**2
    loss = L2Loss
    loss /= N
    loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    # Cross Entropy Loss would have been
    # Cross Entropy loss cares about both correct class and wrong classes as well
    oneHotVectorlog(prediction) for every single prediction

    # NOTE: You never ever predict based on maximum score in loss calculation and gradient backpropagation
    # You only predict based on maximum score in accuracy calculation  
    # for validation sets or test sets.
    predictedClass = np.argmax(predictions, axis=1)
    '''

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    grads['W1'] = np.zeros(W1.shape)
    grads['W2'] = np.zeros(W2.shape)
    grads['b1'] = np.zeros(b1.shape)
    grads['b2'] = np.zeros(b2.shape)
    #############################################################################
    # Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    #----------------------------------------------------------------------------
    # Compute gradient for W2
    #----------------------------------------------------------------------------
    # gradient for W2 is similar to softmax loss implemented earlier
    # Except you replace the input which was X earlier to firstHiddenLayer now
    # and the input output shape needs to be resized accordingly
    num = np.exp(np.dot(firstHiddenLayer, W2) + b2)  # (N, C)
    den = np.sum(num, axis=1) 
    result = num/np.reshape(den, (-1, 1))  # (N, C)
    # Deduct anywhere where the prediction is equal to the correct label
    result[np.arange(N), y] -= 1.0
    # Normalize by number of training 
    # Also, the chain rule multiplies at each step,
    # so doesn't matter if you take division by N first
    # since multiplication and division are commutative
    result /= N # (N, C) 

    # Approach 2: Figured out the gradients are shared before multiplying into them
    # Gradients shared are:
    # dLdZ for dW2 and db2
    # dLdh1 for dW1 and db1
    dLdZ = result # (N, C)
    dLdV = np.dot(dLdZ, np.transpose(W2)) # (N, H)
    # Handle ReLU, ReLU doesn't alter the shape, only stop gradients from propagating.
    # since the gradients would multiply 0
    dLdh1 = dLdV
    dLdh1[np.where(firstHiddenLayer <= 0.0)] = 0.0 # (N, H)

    dW2 = np.zeros(W2.shape) # (H, C)
    db2 = np.zeros(b2.shape) # (C)
    dW1 = np.zeros(W1.shape) # (D, H)
    db1 = np.zeros(b1.shape) # (H)
    #------------------------------------------------
    dW2 += np.dot(np.transpose(firstHiddenLayer), dLdZ)
    dW2 += 2.0 * reg *  W2 # Regularization
    #------------------------------------------------
    # Sum over the training examples for each class in the 2nd bias
    # Since it multiplies by 1 for every class for every training example
    db2 += np.sum(dLdZ, axis = 0)
    #------------------------------------------------
    dW1 += np.dot(np.transpose(X), dLdh1) # (D, H)
    dW1 += 2.0 * reg *  W1 # Regularization
    #------------------------------------------------
    # Sum over the training examples for each hidden unit neuron out of H of them
    db1 += np.sum(dLdh1, axis = 0) 
    #------------------------------------------------
    '''
    Approach 1: Step by step of your first successful attempt at dW2 and db2
                before realizing you could share the gradients
                Sharing gradient computation => Dynamic programming
    Maintain it to be here for future reference. 
    num = np.exp(np.dot(firstHiddenLayer, W2) + b2) 
    den = np.sum(num, axis=1) 
    result = num/np.reshape(den, (-1, 1)) 

    dW2 += np.dot(np.transpose(firstHiddenLayer), result)

    mask = np.zeros((firstHiddenLayer.shape[0], W2.shape[1]))
    mask[np.arange(firstHiddenLayer.shape[0]), y] = 1.0
    dW2 -= np.dot(np.transpose(firstHiddenLayer), mask) 
    dW2 /= N # Still divide by number of training examples
    dW2 += 2.0 * reg *  W2

    #----------------------------------------------------------------------------
    # Compute gradient for b2
    #----------------------------------------------------------------------------
    db2 = np.zeros(b2.shape) #(C)
    num = np.exp(np.dot(firstHiddenLayer, W2) + b2) 
    den = np.sum(num, axis=1) 
    result = num/np.reshape(den, (-1, 1)) 

    db2 += np.sum(result, axis = 0) 

    # Only deduct ones for the correct classes
    mask = np.zeros((N, b2.shape[0]))
    mask[np.arange(firstHiddenLayer.shape[0]), y] = 1.0
    db2 -= np.sum(mask, axis = 0)

    db2 /= N
    # No need regularization for bias since bias has a lot more data, and will be less likely to overfit.
    '''
    #----------------------------------------------------------------------------
    grads['W2'] = dW2
    grads['W1'] = dW1
    grads['b2'] = db2
    grads['b1'] = db1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indices = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[indices]
      y_batch = y[indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2'] 
      self.params['b2'] -= learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X)
    # Won't even need to perform softmax on the scores as it's just the maximum 
    # For the softmax classifier
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################
    return y_pred

def mySoftmax(x):
    num = np.exp(x)
    den = np.sum(num, axis = 1)
    fx = num/np.reshape(den, (-1, 1))
    return fx
