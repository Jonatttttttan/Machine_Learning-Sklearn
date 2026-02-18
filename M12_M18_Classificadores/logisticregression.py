import preprocessing as pre
from MachineLearning.Ml02 import XTrain




def computeLogisticRegressionModel(XTrain, yTrain):
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(solver = 'lbfgs')
    classifier.fit(XTrain[0], yTrain)

    return classifier

def predictModel(classifier, XTest):
    return classifier.predict(XTest[0])

def evaluateModel(classifier, yTest, yPred):
    from sklearn.metrics import confusion_matrix
    confusionMatrix = confusion_matrix(yTest, yPred)

    return confusionMatrix

def computeLogisticRegressionExample(filename):
    X, y, _ = pre.loadDataset(filename, ",")
    X = pre.fillMissingData(X, 2, 3)

    X = pre.computeCategorization(X)
    X = pre.computeCategorization(X)

    XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, 0.15)

    XTrain = pre.computeScaling(XTrain)
    XTest = pre.computeScaling(XTest)

    classifier = computeLogisticRegressionModel(XTrain, yTrain)
    yPred = predictModel(classifier, XTest)

    return evaluateModel(classifier, yTest, yPred)


