"""

Code Challenges:
    Name:
        University Admission Prediction Tool
    File Name:
        uni_admin.py
    Dataset:
        University_data.csv
    Problem Statement:
         Perform Linear regression to predict the chance of admission based on all the features given.
         Based on the above trained results, what will be your estimated chance of admission.

"""



#Importing Libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")

dataset = pd.read_csv("University_data.csv")

features = dataset.iloc[:,0:6].values
labels = dataset.iloc[:,-1].values



from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train,labels_train)


# Predicting the Test set results
labels_pred = regressor.predict(features_test)


print (pd.DataFrame(zip(labels_pred, labels_test)))



#Model accuracy
print (regressor.score(features_test, labels_test))
print (regressor.score(features_train, labels_train))


