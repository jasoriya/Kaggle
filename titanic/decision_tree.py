import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train_data = pd.read_csv("train.csv", sep = ",")
test_data = pd.read_csv("test.csv", sep = ",")
train_data['Embarked'] = train_data['Embarked'].factorize()[0]
train_data['Sex'] = train_data['Sex'].factorize()[0]

X_train = train_data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis =1).values
y_train = train_data.iloc[:, 1].values

from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(X_train[:, 3])
X_train[:, 3] = imputer.transform(X_train[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[7])
X_train = onehotencoder.fit_transform(X_train).toarray()