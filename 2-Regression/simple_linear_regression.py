# Simple linear regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# importing the dataset

dataset = pd.read_csv('/Users/Leo/PycharmProjects/ML-Udemy/2-Regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # X matrix of features - independent variable (Yrs of exp)
y = dataset.iloc[:, 1].values  # dependent variable (Salary)

# Splitting the data into the training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/3, random_state=0)

# Fitting simple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualizing the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set Results)')
plt.xlabel('Years of experience')
plt.ylabel('Salary in $')
plt.show()

# Visualizing the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set Results)')
plt.xlabel('Years of experience')
plt.ylabel('Salary in $')
plt.show()
