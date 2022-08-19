#this is to import numpy
import numpy as np

#ths is to import pandas as pd
import pandas as pd

#this code is to import matplotlib library
import matplotlib.pyplot as plt

Dataset = pd.read_csv("Tonaton.csv")

#creating a variable x and y to store the independent and dependent
x= Dataset.iloc[:,0:2].values
y= Dataset.iloc[:,2:3].values

#Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_changer =make_column_transformer((OneHotEncoder(),[0]),remainder='passthrough')
x = column_changer.fit_transform(x)

import scipy.sparse
#mat = scipy.sparse.eye(3)
x=pd.DataFrame.sparse.from_spmatrix(x)

#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set result
y_pred = regressor.predict(x_test)

#Evaluating the module
from sklearn.metrics import r2_score as r2
r2_score = r2(y_pred,y_test)
print(r2_score*100, '%')
