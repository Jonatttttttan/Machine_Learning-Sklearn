import preprocessing as pre
from MachineLearning.M12_M18_Classificadores.logisticregression import predictModel


def computeKNNModel(XTrain, yTrain, XTest):
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=5, p = 2) #p=2 distância euclidiana
    classifier.fit(XTrain[0], yTrain)

    return classifier

caminho1 = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\titanic.csv"
caminho2 = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\pc.csv"

def predictModel(classifier, XTest):
    return classifier.predict(XTest[0])

def evaluateModel(classifier, yPred, yTest):
    from sklearn.metrics import confusion_matrix
    confusionMatrix = confusion_matrix(yTest, yPred)

    return confusionMatrix

def computeKNNExample(filename):
    X, y, csv = pre.loadDataset(filename, ",")
    X = pre.fillMissingData(X, 2, 3)

    #sex
    X = pre.computeCategorization(X)
    #embark
    X = pre.computeCategorization(X)

    XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, 0.15)
    XTrain = pre.computeScaling(XTrain)
    XTest = pre.computeScaling(XTest)

    classifier = computeKNNModel(XTrain, yTrain, XTest)
    yPred = predictModel(classifier, XTest)
    return evaluateModel(classifier, yPred, yTest)

if __name__ == "__main__":
    print(computeKNNExample())
