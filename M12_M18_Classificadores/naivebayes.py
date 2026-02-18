from classification import ClassificationModel
caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\titanic.csv"
class NaiveBayes(ClassificationModel):
    def computeModel(XTrain, yTrain):
        from sklearn.naive_bayes import GaussianNB

        classifier = GaussianNB()
        classifier.fit(XTrain[0], yTrain)

        return classifier

    def computeExample(selffilename):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(caminho, True)

        classifier = NaiveBayes.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest, 0)
        return ClassificationModel.evaluateModel(yPred, yTest)

if __name__ == "__main__":
    print(NaiveBayes.computeExample(caminho))

