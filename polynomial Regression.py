import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#fitting linear regression
from sklearn.linear_model import LinearRegression
ln_regg1=LinearRegression()
ln_regg1.fit(x,y)
#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
ln_regg2=LinearRegression()
ln_regg2.fit(x_poly,y)
plt.scatter(x,y,color='r')
plt.plot(x,ln_regg1.predict(x))
plt.title("truth or bluff(linear model)")
plt.xlabel("level")
plt.ylabel("salary")
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len (x_grid),1))
plt.scatter(x,y,color='r')
plt.plot(x_grid,ln_regg2.predict(poly_reg.fit_transform(x_grid)))
plt.title("truth or bluff(polymomial model)")
plt.xlabel("level")
plt.ylabel("salary")
plt.show()
#predicting a new result with linear regression
ln_regg1.predict(6.5)
##predicting a new result with polynomial regression
ln_regg2.predict(poly_reg.fit_transform(6.5))
