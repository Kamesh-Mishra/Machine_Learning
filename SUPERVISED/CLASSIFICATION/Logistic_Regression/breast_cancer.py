
"""

Q1. (Create a program that fulfills the following specification.)

Program Specification:

Import breast_cancer.csv file.

This breast cancer database was obtained from the University of Wisconsin

Hospitals, Madison from Dr. William H. Wolberg.

Attribute Information: (class attribute has been moved to last column)

Sample Code Number(id number)                     ----> represented by column A.

Clump Thickness (1 – 10)                                     ----> represented by column B.
Uniformity of Cell Size(1 - 10)                             ----> represented by column C.
Uniformity of Cell Shape (1 - 10)                        ----> represented by column D.
Marginal Adhesion (1 - 10)                                  ----> represented by column E.
Single Epithelial Cell Size (1 - 10)                        ----> represented by column F.
Bare Nuclei (1 - 10)                                               ----> represented by column G.
Bland Chromatin (1 - 10)                                     ----> represented by column H.
Normal Nucleoli (1 - 10)                                      ----> represented by column I.
Mitoses (1 - 10)                                                     ----> represented by column J.
Class: (2 for Benign and 4 for Malignant)         ----> represented by column K. 
A Benign tumor is not a cancerous tumor and Malignant tumor is a cancerous tumor.

                    Impute the missing values with the most frequent values.
                    Perform Classification on the given data-set to predict if the tumor is cancerous or not.
                    Check the accuracy of the model.
                    Predict whether a women has Benign tumor or Malignant tumor, if her Clump thickness is around 6, uniformity of cell size is 2, Uniformity of Cell Shape is 5, Marginal Adhesion is 3, Bland Chromatin is 9, Mitoses is 4, Bare Nuclei is 7, Normal Nuclei is 2 and Single Epithelial Cell Size is 2

(you can neglect the id number column as it doesn't seem  a predictor column)
 
 
 """
 
 
 
 ######################## by logistic regression #################
 
import numpy as np
import pandas as pd
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")

data = pd.read_csv("breast_cancer.csv")

data.isnull().any()


data = data.fillna(data['G'].median())

data.isnull().any()

features = data.iloc[:,1:10].values

labels = data.iloc[:,-1].values
 
 
 

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)

 
 
 
 
 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)


print (pd.DataFrame(zip(labels_pred, labels_test)))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm) 


# Printing Score for the Regression Model
print (classifier.score(features_train,labels_train))




os.chdir("E:/Machine_Learning/SUPERVISED/CLASSIFICATION/Logistic_Regression/")

# Save Model To a File Using Python Pickle
import pickle
with open('logistic_bcancer_model_pickle','wb') as file:
    pickle.dump(classifier,file)
with open('logistic_bcancer_model_pickle','rb') as file:
    mp = pickle.load(file)

x = [[6,2,5,3,2,7,9,2,4]]
print(mp.predict(x))



# Save Trained Model Using joblib
from sklearn.externals import joblib
joblib.dump(classifier, 'logistic_BCancer_model_joblib')

mj = joblib.load('logistic_BCancer_model_joblib')

x = [[6,2,5,3,2,7,9,2,4]]
print(mj.predict(x))

  






# x = [[6,2,5,3,2,7,9,2,4]]
# predd = classifier.predict(x)

if predd == 2:
    Type = 'Benign' 
elif predd == 4:
    Type = 'Malignant'
print("women has class of tumor is : ", Type)
