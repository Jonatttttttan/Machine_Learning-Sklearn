from classification import ClassificationModel
caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\titanic.csv"
class SVM(ClassificationModel):
    def computeModel(XTrain, yTrain, k, d):
        from sklearn.svm import SVC

        classifier = SVC(kernel=k, degree = d)
        classifier.fit(XTrain[0], yTrain)

        return classifier

    def computeExample(filename, kernel, degree):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename, True)

        classifier = SVM.computeModel(XTrain, yTrain, kernel, degree)
        yPred = ClassificationModel.predictModel(classifier, XTest, 0)
        return ClassificationModel.evaluateModel(yPred, yTest)

if __name__ == "__main__":
    print(SVM.computeExample(caminho, "linear", 0))