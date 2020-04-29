

"""

Q1. (Create a program that fulfills the following specification.)
bluegills.csv

How is the length of a bluegill fish related to its age?

In 1981, n = 78 bluegills were randomly sampled from Lake Mary in Minnesota. The researchers (Cook and Weisberg, 1999) measured and recorded the following data (Import bluegills.csv File)

Response variable(Dependent): length (in mm) of the fish

Potential Predictor (Independent Variable): age (in years) of the fish

    How is the length of a bluegill fish best related to its age? (Linear/Quadratic nature?)
    What is the length of a randomly selected five-year-old bluegill fish? Perform polynomial regression on the dataset.

NOTE: Observe that 80.1% of the variation in the length of bluegill fish is reduced by taking into account a quadratic function of the age of the fish.

"""



# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/files")
# Importing the dataset
dataset = pd.read_csv('bluegills.csv')


features = dataset.iloc[:, [0]].values

labels = dataset.iloc[:, [-1]].values




# Visualising the dataset
plt.scatter(features, labels, color = 'red')
plt.title('data distribution')
plt.xlabel('AGE')
plt.ylabel('Length')
plt.show()


#  dataset sorting
dataset = dataset.sort_values("age")

features = dataset.iloc[:, [0]].values

labels = dataset.iloc[:, [-1]].values

# Visualising the dataset
plt.scatter(features, labels, color = 'red')
plt.title('data distribution')
plt.xlabel('AGE')
plt.ylabel('Length')
plt.show()



# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(features, labels)


x = [5]
x = np.array(x)
x = x.reshape(1,-1)

regressor.predict(x)


regressor.score(features,labels)



# Visualising the Linear Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, regressor.predict(features), color = 'blue')
plt.title('data distribution')
plt.xlabel('AGE')
plt.ylabel('Length')
plt.show()












# Fitting Polynomial Regression to the dataset


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/files")
# Importing the dataset
dataset = pd.read_csv('bluegills.csv')

dataset = dataset.sort_values("age")


features = dataset.iloc[:, [0]].values

labels = dataset.iloc[:, [-1]].values





from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
features_poly = poly_reg.fit_transform(features)


poly_reg.fit(features_poly, labels)

print ("Predicting result with Linear Regression")
print (regressor.predict([[5]]))


lin_reg_2 = LinearRegression()
lin_reg_2.fit(features_poly, labels)



print ("Predicting result with Polynomial Regression")
print (lin_reg_2.predict(poly_reg.fit_transform(x)))


# Score for Linear Regression Model
linear_score = regressor.score(features,labels)
print("linear score ",linear_score)

# Score for Polynomial Regression Model
poly_score = lin_reg_2.score(features_poly,labels)
print("polynomial score ",poly_score)












# Visualising the Polynomial Regression results
plt.scatter(features, labels, color = 'red')
plt.plot(features, lin_reg_2.predict(poly_reg.fit_transform(features)), color = 'blue')
plt.title('data distribution')
plt.xlabel('AGE')
plt.ylabel('Length')
plt.show()




# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
features_grid = np.arange(min(features), max(features), 0.1)
features_grid = features_grid.reshape(-1, 1)
plt.scatter(features, labels, color = 'red')
plt.plot(features_grid, lin_reg_2.predict(poly_reg.fit_transform(features_grid)), color = 'blue')
plt.title('data distribution')
plt.xlabel('AGE')
plt.ylabel('Length')
plt.show()





