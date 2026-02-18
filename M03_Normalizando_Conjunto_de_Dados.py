import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

import time

caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\admission.csv"

baseDeDados = pd.read_csv(caminho, delimiter=";")

X = baseDeDados.iloc[:, :-1].values
y = baseDeDados.iloc[:,-1].values



print("Preenchendo dados faltantes")
imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer = imputer.fit_transform(X[:, 1:])
print("ok!")
print(X)

print("Computando rotulação")
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:,0])

X = X[:, 1:]
D = pd.get_dummies(X[:,0])
X = np.insert(X, 0, D.values, axis=1)
print("ok!")
print(X)

print("Separando conjuntos de teste e treino...")
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

print("Computando conjuntos de teste e treino...")
from sklearn.preprocessing import StandardScaler
scaleX = StandardScaler()
Xtrain = scaleX.fit_transform(XTrain)
Xtest = scaleX.fit_transform(XTest)
print("ok!")
print(Xtrain)


