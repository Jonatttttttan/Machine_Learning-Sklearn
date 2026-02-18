import numpy as np
import pandas as pd


caminho = "C:\\Users\\Nitro\\Desktop\\Redes-Neurais\\svbr_Aula_1ML.csv"

baseDeDados = pd.read_csv(caminho, delimiter=';')
X  =baseDeDados.iloc[:,:].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer = imputer.fit(X[:,1:3])
X = imputer.transform(X[:,1:3]).astype(str)
X = np.insert(X, 0, baseDeDados.iloc[:,0].values, axis=1) # Axis 1 coluna | Axis 0 linha

print(X)
