

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


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


dataset = pd.read_csv("Bahubali2_vs_Dangal.csv")

 #for bahubali

labels = dataset.iloc[:,[1]].values

features = dataset.iloc[:,[0]].values


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features,labels)

x = [10]
x = np.array(x)
x = x.reshape(1,-1)

y=regressor.predict(x)

#regressor.score

regressor.score(features,labels)




# Now with test and train data 

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2,
                random_state = 0)

regressor = LinearRegression()
regressor.fit(features_train, labels_train)


# Predicting the Test set results
labels_pred = regressor.predict(features_test)


print (pd.DataFrame(zip(labels_pred, labels_test)))



# Visualising the Training set results
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('box office collection')
plt.xlabel('DAY')
plt.ylabel('COLLECTION')
plt.show()


# Visualising the Test set results
#plt.scatter(features_train, labels_train, color = 'green')
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('box office collection')
plt.xlabel('DAY')
plt.ylabel('COLLECTION')
plt.show()


#http://setosa.io/ev/ordinary-least-squares-regression/
# https://medium.com/@ml.at.berkeley/machine-learning-crash-course-part-1-9377322b3042

#Show the animation using learning rate, cost functions and best fit line
#https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9





#Model accuracy
print (regressor.score(features_test, labels_test))
print (regressor.score(features_train, labels_train))
















labels2 = dataset.iloc[:,[2]].values

features2 = dataset.iloc[:,[0]].values


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features2,labels2)

x2 = [10]
x2 = np.array(x)
x2 = x.reshape(1,1)

y2=regressor.predict(x2)


regressor.score(features2,labels2)


# Now with test and train data 

from sklearn.model_selection import train_test_split

features_train2, features_test2, labels_train2, labels_test2 = train_test_split(features2, labels2, test_size = 0.2,
                random_state = 0)

regressor = LinearRegression()
regressor.fit(features_train2, labels_train2)


# Predicting the Test set results
labels_pred = regressor.predict(features_test2)





# Visualising the Training set results
plt.scatter(features_train2, labels_train2, color = 'red')
plt.plot(features_train2, regressor.predict(features_train2), color = 'blue')
plt.title('box office collection')
plt.xlabel('DAY')
plt.ylabel('COLLECTION')
plt.show()


# Visualising the Test set results
#plt.scatter(features_train, labels_train, color = 'green')
plt.scatter(features_test2, labels_test2, color = 'red')
plt.plot(features_train2, regressor.predict(features_train2), color = 'blue')
plt.title('box office collection')
plt.xlabel('DAY')
plt.ylabel('COLLECTION')
plt.show()




print (regressor.score(features_test2, labels_test2))
print (regressor.score(features_train2, labels_train2))



x2 = [10]
x2 = np.array(x2)
x2 = x.reshape(1,1)

y2=regressor.predict(x2)




