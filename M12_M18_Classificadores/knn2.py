from classification import ClassificationModel
caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\titanic.csv"
class KNN(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.neighbors import KNeighborsClassifier

        classifier = KNeighborsClassifier(n_neighbors=5, p = 2)
        classifier.fit(XTrain[0], yTrain)

        return classifier

    def computeExample(filename):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename, True)

        classifier = KNN.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest, 0)
        return ClassificationModel.evaluateModel(yPred, yTest)

if __name__ == "__main__":
    print(KNN.computeExample(caminho))