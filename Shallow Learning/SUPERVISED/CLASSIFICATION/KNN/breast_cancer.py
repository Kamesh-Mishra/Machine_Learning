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
 

 

# Importing the libraries  
import numpy as np  
import matplotlib.pyplot as mtp  
import pandas as pd  
import os 



os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")
data = pd.read_csv("breast_cancer.csv")

data.isnull().any()

data = data.fillna(data['G'].median())

data.isnull().any()

x = data.iloc[:,1:10].values

y = data.iloc[:,-1].values



# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)  





from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) #When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
classifier.fit(x_train, y_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(x_test)

# Predicting the class labels
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)




from sklearn.metrics import accuracy_score  
print (accuracy_score(y_test, y_pred)*100)


os.chdir('E:/Machine_Learning/SUPERVISED/CLASSIFICATION/KNN/')
# saving model as pickle file

import pickle

with open('model_pickle','wb') as f:
    pickle.dump(classifier,f)


with open('model_pickle','rb') as f:
    mp = pickle.load(f)


x = [[6,2,5,3,2,7,9,2,4]]
predd = mp.predict(x)
print(predd)


# x = [[6,2,5,3,2,7,9,2,4]]
# predd = classifier.predict(x)



if predd == 2:
    Type = 'Benign' 
elif predd == 4:
    Type = 'Malignant'
print("women has class of tumor is : ", Type)

