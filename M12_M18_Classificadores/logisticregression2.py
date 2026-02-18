from classification import ClassificationModel
caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\titanic.csv"

class LogisticRegression(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(solver='lbfgs')
        classifier.fit(XTrain[0], yTrain)

        return classifier

    def computeExample(filename):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename, True)

        classifier = LogisticRegression.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest, 0)
        return ClassificationModel.evaluateModel(yPred, yTest)

if __name__ == "__main__":
    print(LogisticRegression.computeExample(caminho))

