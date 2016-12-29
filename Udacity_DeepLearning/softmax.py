import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    # For a 2D matrix
    # Axis = 0 means sum across each columns
    return np.exp(x)/np.sum(np.exp(x), axis=0)

# Values {-2.0, -1.9, ... , 5.9, 6.0}
x = np.arange(-2.0, 6.0, 0.1)
 
# Stack the values of x on top of a bunch of 1's with same size as x
# First row is x, 2nd row is 1's
# [
#  [-2,..., 6]
#  [1, ..., 1]
# ]
scores = np.vstack([x, np.ones_like(x)])
# Plot x against with the transpose of the softmax scores.  print softmax(scores)
# Transpose it as you summed across each columns
# And to plot, you need to give it row by row for each point.
plt.plot(x, softmax(scores).T, linewidth=2)
# Note: That at every x, the sum between points on both graphs will equal 1
plt.show()

# When multiply all scores by a scalar > 1,
# probabilities gets closer to  0 for smaller values and 1 for larger values
# as the distances between the scores increases

plt.plot(x, softmax(scores*10).T, linewidth=2)
# Note: That at every x, the sum between points on both graphs will equal 1
plt.show()

# When divide all scores by a scalar > 1, 
# probabilities gets closer to the uniform distribution
# as the distance between the scores decreases.

plt.plot(x, softmax(scores/10.0).T, linewidth=2)
# Note: That at every x, the sum between points on both graphs will equal 1
plt.show()

# Therefore, increaseing size of outputs will increase confidence of classifier.
