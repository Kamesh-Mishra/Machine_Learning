"""


CODE CHALLENGES

Q1. (Create a program that fulfills the following specification.)
Foodtruck.csv


You will implement linear regression to predict the profits for a food chain company.

Case: Suppose you are the CEO of a restaurant franchise and are considering
different cities for opening a new outlet.
The chain already has food-trucks in various cities and you have data for profits
and populations from the cities.
You would like to use this data to help you select which city to expand to next.

Foodtruck.csv contains the dataset for our linear regression problem.
The first column is the population of a city and the second column is the
profit of a food truck in that city.
A negative value for profit indicates a loss.

Perform Simple Linear regression to predict the profit based on the population
observed and visualize the result.
Based on the above trained results, what will be your estimated profit,
if you set up your outlet in Jaipur? (Current population in Jaipur is 3.073 million)

"""







# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")

# Importing the dataset
dataset = pd.read_csv('Foodtruck.csv')

#  dataset.plot.scatter(x= "Population", y ="Profit")


features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, [-1]].values



# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()      #constructor making

regressor.fit(features, labels)

regressor.score(features, labels)

regressor.predict(np.array([3.073]).reshape(1,-1))



# Visualising the Training set results
plt.scatter(features, labels, color = 'red')
plt.plot(features, regressor.predict(features), color = 'blue')
plt.title('Income vs ML-Experience (Training set)')
plt.xlabel('ML-Experience')
plt.ylabel('Profit')
plt.show()
