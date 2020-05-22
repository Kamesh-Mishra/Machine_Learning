

"""
Q1. dataset: 
    pima-indians-diabetes.csv

Perform both Multinomial and Gaussian Naive Bayes classification 
after taking care of NA values (maybe replaced with zero in dataset)

Calculate accuracy for both Naive Bayes classification model.

"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as mtp
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")
df = pd.read_csv("pima-indians-diabetes.csv", names = ['A','B','C','D','E','F','G','H','I'])


df.isnull().any()

feature = df.iloc[:,:-1]

for i in feature.columns:                        # filling 0 with mode value of column
        
    feature[i]=feature[i].replace(0,feature[i].mean())

feature = feature.values
labels = df.iloc[:,[-1]].values

  
from sklearn.model_selection import train_test_split
feature_train, feature_test, labels_train, labels_test = train_test_split(feature, labels, test_size= .3, random_state = 6)


#####################################################################


# GaussianNB


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(feature, labels)

labels_pred = model.predict(feature_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

score = model.score(feature_test, labels_test)
print(score)


from sklearn import metrics
accuracy=  metrics.accuracy_score(labels_test, labels_pred)
print(accuracy)


######################################################################



# Feature Scaling  
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
feature_train = sc.fit_transform(feature_train)  
feature_test = sc.transform(feature_test)  



from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(feature, labels)

labels_pred = model.predict(feature_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(labels_test, labels_pred)
print(cm1)

score1 = model.score(feature_test, labels_test)
print(score1)


from sklearn import metrics
accuracy1 = metrics.accuracy_score(labels_test, labels_pred)
print(accuracy1)









# with StandardScaling 



# # Feature Scaling  
# from sklearn.preprocessing import StandardScaler  
# sc = StandardScaler()  
# feature_train = sc.fit_transform(feature_train)  
# feature_test = sc.transform(feature_test)  



# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB()
# model.fit(feature, labels)

# labels_pred = model.predict(feature_test)

# from sklearn.metrics import confusion_matrix
# cm2 = confusion_matrix(labels_test, labels_pred)
# print(cm2)

# score2 = model.score(feature_test, labels_test)
# print(score2)


# from sklearn import metrics
# accuracy2 = metrics.accuracy_score(labels_test, labels_pred)
# print(accuracy2)


#####################################################################


print("Comaprison : accuracy of GaussianNB({}) is greater than MultinomialNB({}) ".format(accuracy, accuracy1),)



