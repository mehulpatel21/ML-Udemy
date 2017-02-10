# Polynomial Regression

# importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset

dataset = pd.read_csv('/Users/Leo/PycharmProjects/ML-Udemy/4-Polynomial_Linear_Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the data into the training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
