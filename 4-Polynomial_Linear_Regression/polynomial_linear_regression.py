# Polynomial Regression

# importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


# importing the dataset

dataset = pd.read_csv('/Users/Leo/PycharmProjects/ML-Udemy/4-Polynomial_Linear_Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear regression to dataset
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)


# Fitting Polynomial Linear Regression to dataset
polynomialRegressor = PolynomialFeatures(degree=4)
X_poly = polynomialRegressor.fit_transform(X)
# instead of using the 1st degree features, we used X_poly.
linearRegressor2 = LinearRegression()
linearRegressor2.fit(X_poly, y)

# Visualizing the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, linearRegressor.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression Results)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualizing the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linearRegressor2.predict(polynomialRegressor.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Linear Regression Results)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
linearRegressor.predict(6.5)

# Predicting a new result with Polynomial Linear Regression
linearRegressor2.predict(polynomialRegressor.fit_transform(6.5))
