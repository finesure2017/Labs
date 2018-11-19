'''
Question 1 Skeleton Code
Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc # For scipy.misc.logsumexp()

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class.

    Should return a numpy array of size (10,64)
    The i_th row will correspond to the mean estimate for digit class i

    train_data is (700, 64). 700 training images, each image is 8x8 pixels
    train_labels is (700, 1). Each represent a supervised label for each training image
    The labels can be one of 10 classes {0, 1, 2, ..., 9}
    '''
    numClass = 10
    dimension = train_data.shape[1]
    means = np.zeros((numClass, dimension))
    for currClassIndex in range(numClass):
        indices = np.where(train_labels == currClassIndex)
        currClassTrainData = train_data[indices]
        means[currClassIndex] = np.mean(currClassTrainData, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    means = compute_mean_mles(train_data, train_labels)
    numClass = 10
    dimension = train_data.shape[1]
    covariances = np.zeros((numClass, dimension, dimension))
    for currClassIndex in range(numClass):
        indices = np.where(train_labels == currClassIndex)
        currClassTrainData = train_data[indices]
        numTrainData = currClassTrainData.shape[0]
        difference = currClassTrainData - means[currClassIndex]
        currCovariance = np.dot(np.transpose(difference), difference)/float(numTrainData)
        # note: Plot looks better if you don't add 0.01
        # note: Train and test accuracy much higher if you do
        # For numerical stability
        currCovariance += (0.01 * np.identity(dimension))
        covariances[currClassIndex] = currCovariance
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    digits are X
    Should return an n x 10 numpy array 
    '''
    numData = digits.shape[0]
    dimension = digits.shape[1]
    X = digits
    numClass = 10
    generativeLogLikelihood = np.zeros((numData, numClass))
    constantWrtEachClass = float(dimension) * np.log(2.0 * np.pi) 
    # Note: Below is wrong as you sum over all training points for a class
    # resuling in (1, C) NOT (N, C), C = numClass, N = numData
    # Calculate probability with respect to each class
    # note: Just use for loop since only over 10 classes, so won't be much slower
    for currClassIndex in range(numClass):
        currClassMean = means[currClassIndex]
        currClassCovariance = covariances[currClassIndex]
        logDetCov = np.log(np.linalg.det(currClassCovariance))
        constWrtCurrClass = logDetCov
        invCov = np.linalg.inv(currClassCovariance)
        difference = (X - currClassMean)
        # Pick out diagonal entries instead of trace
        # np.einsum() does so without any redundant calculation
        mahalanobisDistance = np.einsum('ij,jk,ki->i', difference, invCov, np.transpose(difference))
        currClassGenerativeLogLikelihood = (-0.5) * (constantWrtEachClass + constWrtCurrClass + mahalanobisDistance)
        generativeLogLikelihood[:, currClassIndex] = currClassGenerativeLogLikelihood
    return generativeLogLikelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:
        log p(y|x, mu, Sigma)

    digits are X

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    numData = digits.shape[0]
    dimension = digits.shape[1]
    X = digits
    numClass = 10
    # Since the class over labels are uniform, they cancel out
    # Hence, below is the entire numerator
    generativeLogLikelihood = generative_likelihood(digits, means, covariances)
    numerator = generativeLogLikelihood
    denominator = np.sum(np.exp(generativeLogLikelihood), axis=1)
    conditionalLogLikelihood = np.zeros((numData, numClass))
    conditionalLogLikelihood = numerator - np.reshape(np.log(denominator), (-1, 1))
    return conditionalLogLikelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels
        AVG( log p(y_i|x_i, mu, Sigma) )
    digits are X
    labels are y
    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    conditionalLogLikelihood = conditional_likelihood(digits, means, covariances)
    labels = labels.astype(int)
    conditionalLogLikelihoodTrueClass = conditionalLogLikelihood[np.arange(len(conditionalLogLikelihood)), labels]
    averageConditionalLogLikelihoodTrueClass = np.mean(conditionalLogLikelihoodTrueClass)
    return averageConditionalLogLikelihoodTrueClass

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    # Since denominator is a constant, it doesn't affect prediction
    # Since prior over classes is uniform, it doesn't affect prediction either
    # Hence, use generative logLikelihood directly instead
    generativeLogLikelihood = generative_likelihood(digits, means, covariances)
    pred =  np.argmax(generativeLogLikelihood, axis=1)
    return pred

def computeAccuracy(labels, predictions):
    '''
    Compute accuracy
    '''
    # Match is number of 0s
    notMatch = labels - predictions
    numCorrect = len(np.where(notMatch == 0)[0])
    accuracy = numCorrect/float(labels.size)
    return accuracy

def plotAndSaveLeadingEigenvector(covariances):
    fig = plt.figure()
    for currCovarianceIndex in range(covariances.shape[0]):
        currCovariance = covariances[currCovarianceIndex] # (D, D)
        # D eigenvalues, D eigenvectors, each eigenvector is dimension D
        eigenValues, eigenVectors = np.linalg.eig(currCovariance)
        largestEigenvector = eigenVectors[:, eigenValues.argmax()]
        ax1 = fig.add_subplot(2, 5,currCovarianceIndex + 1)
        ax1.imshow(np.reshape(largestEigenvector, (8, 8)), cmap='gray')
    plt.savefig("leadingEigenvectorCovarianceDigits.png")

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    # 700 for each digit, total of 7000
    print("Train data shape: ", train_data.shape)
    print("Train labels shape: ", train_labels.shape)

    # 400 for each digit, total of 4000
    print("Test data shape: ", test_data.shape)
    print("Test labels shape: ", test_labels.shape)  # Values are in {0, 1, ..., 9}

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # Average Conditional Log Likelihood
    trainAverageConditionalLogLikelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    testAverageConditionalLogLikelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("Train Average Conditional Log Likelihood:", trainAverageConditionalLogLikelihood)
    print("Test Average Conditional Log Likelihood:", testAverageConditionalLogLikelihood)

    # Ensure probabilities close to 1.0
    print("Train Average Conditional Likelihood:", np.exp(trainAverageConditionalLogLikelihood))
    print("Test Average Conditional Likelihood:", np.exp(testAverageConditionalLogLikelihood))

    # Accuracy
    trainPrediction = classify_data(train_data, means, covariances)
    trainAccuracy = computeAccuracy(train_labels, trainPrediction)
    testPrediction = classify_data(test_data, means, covariances)
    testAccuracy = computeAccuracy(test_labels, testPrediction)
    print("Train Accuracy: ", trainAccuracy)
    print("Test Accuracy: ", testAccuracy)

    # Plot 8 by 8 images of all 10 leading eigenvectors
    # Save it as leading eigenvector plot 
    plotAndSaveLeadingEigenvector(covariances)

if __name__ == '__main__':
    main()
