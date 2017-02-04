# Simple linear regression

import numpy as np
import pandas as pd
import matplotlib
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# importing the dataset

dataset = pd.read_csv('/Users/Leo/PycharmProjects/ML-Udemy/2-Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # X matrix of features - independent variable (Yrs of exp)
y = dataset.iloc[:, 1].values  # dependent variable (Salary)

# Splitting the data into the training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature Scaling

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
