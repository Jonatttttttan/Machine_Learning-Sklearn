import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\svbr.csv"

def loadDataset(filename):
    baseDeDados = pd.read_csv(filename, delimiter=";")
    X = baseDeDados.iloc[:, :-1].values
    y = baseDeDados.iloc[:,-1].values
    return X, y

def fillMissingData(X):
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    X[:, 1:] = imputer.fit_transform(X[:, 1:])
    return X


def computeCategorization(X):
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:,0])

    # one hot encoding
    D = pd.get_dummies(X[:,0])
    X = X[:, 1:]
    X = np.insert(X, 0, D.values, axis=1)
    return X


def splitTrainTestSets(X, y, testSize):

    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)
    return XTrain, XTest, yTrain, yTest

def computeScaling(train, test):

    from sklearn.preprocessing import StandardScaler
    scaleX = StandardScaler()
    train = scaleX.fit_transform(train)
    test = scaleX.fit_transform(test)
    return train, test

def computeLinearRegressionModel(XTrain, yTrain):
    from sklearn.linear_model import  LinearRegression
    regressor = LinearRegression()
    regressor.fit(XTrain, yTrain)
    
    #yPred = regressor.predict(XTest)
    return regressor

    # Gerar gráfico
    '''
    import matplotlib.pyplot as plt
    plt.scatter(XTest[:,-1], yTest, color='red')
    plt.plot(XTest[:, -1], regressor.predict(XTest), color='blue')
    plt.title("Inscritos x Visualizações (SVBR)")
    plt.xlabel("Total de Inscritos")
    plt.ylabel("Total de Visualizações")
    plt.show()'''

def showPlot(X, y, linearRegressor):
    import matplotlib.pyplot as plt

    plt.scatter(X, y, color="red")
    plt.plot(X, linearRegressor.predict(X), color="blue")
    plt.title("Comparando pontos reais com a reta produzida pela regressão linear")
    plt.xlabel("Experiência em anos")
    plt.ylabel("Salário")
    plt.show()


def runLinearRefressionExample(filename):
    X, y = loadDataset(filename)
    X = fillMissingData(X)
    X = computeCategorization(X)
    XTrain, XTest, yTrain,yTest = splitTrainTestSets(X, y, 0.8)
    computeLinearRegressionModel(XTrain, yTrain)


if __name__ == "__main__":
    runLinearRefressionExample(caminho)
