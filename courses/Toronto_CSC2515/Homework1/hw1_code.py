import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import graphviz # To visualize decision tree
import math

def load_data():
    '''
    This function loads the data
    '''
    cleanFakeFileName = 'clean_fake.txt'
    cleanRealFileName = 'clean_real.txt' 
    trainPercentage = 0.7
    validationPercentage = 0.15
    testPercentage = 0.15

    cleanFakeFile = open(cleanFakeFileName)
    cleanRealFile = open(cleanRealFileName)
    allFakeLines = cleanFakeFile.read().splitlines()
    allRealLines = cleanRealFile.read().splitlines()
    cleanFakeFile.close()
    cleanRealFile.close()
    listOfSentences = allFakeLines + allRealLines
    allFakeLines = np.array(allFakeLines)
    allRealLines = np.array(allRealLines)

    countVec = CountVectorizer()
    countVec.fit(listOfSentences)

    numFake = len(allFakeLines)
    numReal = len(allRealLines)
    fakeIndex = np.random.permutation(numFake)
    realIndex = np.random.permutation(numReal)
    trainFakeLines = allFakeLines[fakeIndex[:int(round(trainPercentage*numFake))]]
    trainRealLines = allRealLines[realIndex[:int(round(trainPercentage*numReal))]]
    validFakeLines = allFakeLines[fakeIndex[int(round(trainPercentage*numFake)):int(round((trainPercentage+validationPercentage)*numFake))]]
    validRealLines = allRealLines[realIndex[int(round(trainPercentage*numReal)):int(round((trainPercentage+validationPercentage)*numReal))]]
    testFakeLines = allFakeLines[fakeIndex[int(round((trainPercentage+validationPercentage)*numFake)):]]
    testRealLines = allRealLines[realIndex[int(round((trainPercentage+validationPercentage)*numReal)):]]
    trainLines = np.concatenate((trainFakeLines, trainRealLines))
    validLines = np.concatenate((validFakeLines, validRealLines))
    testLines = np.concatenate((testFakeLines, testRealLines))
    # Term Document Matrix
    xTrain = countVec.transform(trainLines)
    xValid = countVec.transform(validLines)
    xTest = countVec.transform(testLines)
    # 0 => Fake, 1 => Real
    yTrain = np.concatenate((np.zeros(trainFakeLines.shape[0]), np.ones(trainRealLines.shape[0])))
    yValid = np.concatenate((np.zeros(validFakeLines.shape[0]), np.ones(validRealLines.shape[0])))
    yTest = np.concatenate((np.zeros(testFakeLines.shape[0]), np.ones(testRealLines.shape[0])))
    # Shuffle the pairs
    trainShuffleIndex = np.random.permutation(xTrain.shape[0])
    xTrain = xTrain[trainShuffleIndex, :]
    yTrain = yTrain[trainShuffleIndex]
    validShuffleIndex = np.random.permutation(xValid.shape[0])
    xValid = xValid[validShuffleIndex, :]
    yValid = yValid[validShuffleIndex]
    testShuffleIndex = np.random.permutation(xTest.shape[0])
    xTest = xTest[testShuffleIndex, :]
    yTest = yTest[testShuffleIndex]
    return  xTrain, yTrain, xValid, yValid, xTest, yTest, countVec

def select_model(xTrain, yTrain, xValid, yValid, xTest, yTest):
    maxDepthValues = [5, 10, 100, 200, None]  # None means no limit to depth
    criterions = ['gini', 'entropy'] # Entropy for information gain
    maxValidAccuracy = 0.0
    bestCriterion = -1
    bestDepth = -1
    for crite in criterions:
        for maxDepth in maxDepthValues:
            dtc = tree.DecisionTreeClassifier(criterion=crite, max_depth=maxDepth)
            dtc.fit(xTrain, yTrain)
            yValidPred = dtc.predict(xValid)
            validAccuracy = accuracy_score(yValid, yValidPred)
            if validAccuracy > maxValidAccuracy:
                maxValidAccuracy = validAccuracy
                bestDepth = maxDepth
                bestCriterion = crite
            print("criterion = {}, depth = {}, validationAccuracy = {}".format(crite, maxDepth, validAccuracy))
    # Calculate test on the best model
    dtc = tree.DecisionTreeClassifier(criterion=crite, max_depth=maxDepth)
    dtc.fit(xTrain, yTrain)
    yTestPred = dtc.predict(xTest)
    testAccuracy = accuracy_score(yTest, yTestPred)
    print("bestCriterion = {}, bestDepth = {}, maxValidAccuracy = {}, testAccuracy = {}".format(bestCriterion, bestDepth, maxValidAccuracy, testAccuracy))
    return dtc

def visualize_tree(decisionTreeClassifier, countVectorizer, outFileName="treeOutput.dot"):
    # Reference: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
    keys = list(countVectorizer.vocabulary_)
    featureNames = []
    # Form the vocabulary in order
    for i in range(len(keys)):
        currKey = keys[list(countVectorizer.vocabulary_.values()).index(i)]
        featureNames.append(currKey)
    with open(outFileName, "w") as f:
            f = tree.export_graphviz(decisionTreeClassifier, out_file=f, max_depth=2, feature_names=featureNames, class_names=['fake', 'real'])
    return

def entropy(p):
    if p == 0.0:
        # Can't take log of 0
        return 0.0
    return p * math.log(p, 2)

def get_items(s):
        s_coo = s.tocoo()
        return set(zip(s_coo.row, s_coo.col))

def compute_information_gain(wordToSplit, xTrain, yTrain, countVectorizer):
    '''
    Y is the random variable signifying if the headline is fake
    xi is the keyword chosen for the split
    This splits on the first node
    '''
    # Get index of the word to split
    wordToSplitIndex = countVectorizer.vocabulary_[wordToSplit]

    # Get the splits based on word first
    totalNum = yTrain.shape[0]
    numReal = np.sum(yTrain)
    numFake = totalNum - numReal

    totalNumLeft = 0
    numRealLeft = 0
    numFakeLeft = 0
    totalNumRight = 0
    numRealRight = 0
    numFakeRight = 0
    # Loop through each document and count
    for i in range(totalNum):
        # If current paragraph contains the word
        if ((i, wordToSplitIndex) in get_items(xTrain)): 
            totalNumRight += 1
            if yTrain[i] == 1:
                numRealRight += 1
            else:
                numFakeRight += 1
        else: 
            totalNumLeft += 1
            if yTrain[i] == 1:
                numRealLeft += 1
            else:
                numFakeLeft += 1

    # Information gain = H(Y) - (p(left)H(Y|left) + p(right)H(Y|right))
    rootEntropy = - entropy(numReal/float(totalNum)) - entropy(numFake/float(totalNum)) # H(Y)
    leftProb = totalNumLeft/float(totalNum) # p(left)
    rightProb = totalNumRight/float(totalNum) # p(right)
    leftEntropy = - entropy(numRealLeft/float(totalNumLeft)) - entropy(numFakeLeft/float(totalNumLeft)) # H(Y|left)
    rightEntropy = - entropy(numRealRight/float(totalNumRight)) - entropy(numFakeRight/float(totalNumRight)) # H(Y|right)
    infoGain = rootEntropy - ((leftProb * leftEntropy) + (rightProb * rightEntropy))
    return infoGain

if __name__ == '__main__':
    xTrain, yTrain, xValid, yValid, xTest, yTest, countVectorizer = load_data()
    # Get the best DecisionTreeClassifier trained
    dtc = select_model(xTrain, yTrain, xValid, yValid, xTest, yTest)
    visualize_tree(dtc, countVectorizer)
    words = ['trump', 'donald', 'hillary', 'the', 'trumps', 'here']
    for word in words:
        infoGain = compute_information_gain(word, xTrain, yTrain, countVectorizer)
        print("Word = {}, informationGain = {}".format(word, infoGain))
