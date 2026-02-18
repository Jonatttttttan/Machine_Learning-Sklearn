import numpy as np
import pandas as pd

caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\admission.csv"
baseDeDados = pd.read_csv(caminho, delimiter=";")
X = baseDeDados.iloc[:, :-1].values
y = baseDeDados.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer = imputer.fit(X[:,1:])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder() # Rotulos decimais aleatorios para qualitativas
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
print(X)

D = pd.get_dummies(X[:,0])
X = X[:, 1:]

X = np.insert(X, 0, D.values, axis=1) # Label incode matricial
print(X)
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2)
print(f"teste - {XTest.shape}")




