# Simple Linear Regression


"""
Explain the Income Data fields

Divide into features and labels with explanation 

HR Tool to offer a salary to new candidate of 15 yrs

Explain the plotting of two list with range data

Explain best fit line concept, Calculate slope and constant  
y = mx + c 

Gradient Descent concept to find best fit line
to minimise the loss or cost function

For Linear Regression the cost function = Mean Square error
its diffetent for different algorithms 




"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/files")
# Importing the dataset
dataset = pd.read_csv('Income_Data.csv')

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values



# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(features, labels)

x = [12]

x = np.array(x)
x = x.reshape(1,1)
regressor.predict(x)


"""
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2,random_state = 0)

"""


# Now with test and train data 

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2,
                random_state = 0)

regressor = LinearRegression()
regressor.fit(features_train, labels_train)


# Predicting the Test set results
labels_pred = regressor.predict(features_test)


print (pd.DataFrame(labels_pred, np.round(labels_test,2)))




# Visualising the Training set results
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Training set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()


# Visualising the Test set results
#plt.scatter(features_train, labels_train, color = 'green')
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Test set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()


#http://setosa.io/ev/ordinary-least-squares-regression/
# https://medium.com/@ml.at.berkeley/machine-learning-crash-course-part-1-9377322b3042

#Show the animation using learning rate, cost functions and best fit line
#https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9





#Model accuracy
print (regressor.score(features_test, labels_test))
print (regressor.score(features_train, labels_train))

regressor.predict(x)


# Now lets talk about two terms - overfitting and underfitting the data
# best reference available is 
# https://medium.com/towards-data-science/train-test-split-and-cross-validation-in-python-80b61beca4b6


# Do we have case of underfitting?
# Do we have case of overfitting?


"""
If the training score is POOR and test score is POOR then its underfitting

If the training score is GOOD and test score is POOR then its overfitting
"""


# Explain the logic of Best Teacher 100 question story


"""
# Underfitting = no padai

It means that the model does not fit the training data and therefore misses 
the trends in the data
this is usually the result of a very simple model (not enough predictors/independent variables).
"""
 


"""
# Overfitting = ratoo tota

This model will be very accurate on the training data but will probably be very 
not accurate on untrained or new data

This usually happens when the model is too complex (i.e. too many features/variables 
compared to the number of observations). 

It is because this model is not generalized 
Basically, when this happens, the model learns or describes the “noise” in the 
training data instead of the actual relationships between variables in the data.

"""



"""
Solution to Underfitting
    Increase Training Data
    Change the Model from simpler to Complex 
    

Solution to Overfitting
 There are two types of regularization as follows:

    L1 Regularization or Lasso Regularization
    L2 Regularization or Ridge Regularization
    Elastic Net is hybrid of both L1 and L2
"""




"""
Code Challenge
  Name: 
    Food Truck Profit Prediction Tool
  Filename: 
    Foodtruck.py
  Dataset:
    Foodtruck.csv
  Problem Statement:
    Suppose you are the CEO of a restaurant franchise and are considering 
    different cities for opening a new outlet. 
    
    The chain already has food-trucks in various cities and you have data for profits 
    and populations from the cities. 
    
    You would like to use this data to help you select which city to expand to next.
    
    Perform Simple Linear regression to predict the profit based on the 
    population observed and visualize the result.
    
    Based on the above trained results, what will be your estimated profit, 
    
    If you set up your outlet in Jaipur? 
    (Current population in Jaipur is 3.073 million)
        
  Hint: 
    You will implement linear regression to predict the profits for a 
    food chain company.
    Foodtruck.csv contains the dataset for our linear regression problem. 
    The first column is the population of a city and the second column is the 
    profit of a food truck in that city. 
    A negative value for profit indicates a loss.
"""



"""
Code Challenge
  Name: 
    Box Office Collection Prediction Tool
  Filename: 
    Bahubali2vsDangal.py
  Dataset:
    Bahubali2vsDangal.csv
  Problem Statement:
    It contains Data of Day wise collections of the movies Bahubali 2 and Dangal 
    (in crores) for the first 9 days.
    
    Now, you have to write a python code to predict which movie would collect 
    more on the 10th day.
  Hint:
    First Approach - Create two models, one for Bahubali and another for Dangal
    Second Approach - Create one model with two labels
"""




# SIMPLE LINEAR REGRESSION


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/files")
# Importing the dataset
dataset = pd.read_csv('Income_Data.csv')

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

"""
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test= train_test_split(features, labels, test_size = 0.2,random_state = 0)

"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(features, labels)
regressor.score(features, labels)
regressor.predict(np.array([6.5]).reshape(1,-1))

# Predicting the Test set results
labels_pred = regressor.predict(features)

# print (regressor.predict(6.5))

# Visualising the Training set results
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Training set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()


# Visualising the Test set results
#plt.scatter(features_train, labels_train, color = 'green')
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Test set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()



#Model accuracy
print (regressor.score(features_test, labels_test))
print (regressor.score(features_train, labels_train))

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



Q2. (Create a program that fulfills the following specification.)

Import Bahubali2vsDangal.csv file.

It contains Data of Day wise collections of the movies Bahubali 2 and Dangal (in crores) for the first 9 days.
Now, you have to write a python code to predict which movie would collect more on the 10th day.
"""
