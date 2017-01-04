import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    x = matrix
    note: 1xn matrix = array

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    
    if len(x.shape) > 1:
        # Get the maximum across all  everything from the 1th dimension (starting from 0)

        # For example, 2D Matrix means there are 2 dimensions
        # so maximum across 0th dimension is a single number

        '''
        Nesting levels
            axis = 0 => First '[' 
            axis = 1 => Second '['
            Thus, need to always go to deepest axis
        x = [ 
            [a b]
            [c d] 
            ] 
        '''
        # Maximum across 1st dimension is an array
        # with the maximum of each individual 2nd dimension onwards
        tmp = np.max(x, axis = 1)
        print tmp
        # Numpy's broadcasting is in effect in the reshape below
        # For each column in 2D matrix x, every cell in each column
        # is deducting the same tmp variable that was re-shaped
        '''
        x = [             tmp = [
            [a b]    -           max(a,b)
            [c d]                max(c,d)
            ]                   ] 

        = 
        x = [ 
            [(a-max(a,b)) (b-max(a,b))]
            [(c-max(c,d)) (d-max(c,d))] 
            ] 
            
        '''
        x -= tmp.reshape((x.shape[0], 1)) # x.shape = (2,2)
        x = np.exp(x)
        # Divide by the sum across each nested sub array
        tmp = np.sum(x, axis = 1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    
    ### END YOUR CODE
    
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    # Shifting by a constant  + 998 shouldn't change the probabilities 
    # since softmax is invariance to shift
    # from 3,4 to 1001, 1002
    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    # The bigger number (less negative) number will be higher probability 
    # because you are deducting the most negative (smallest) number for regularization
    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
