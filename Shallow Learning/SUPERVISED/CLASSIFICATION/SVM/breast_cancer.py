
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
 


########## poly kernel ########




# Importing the libraries  
import numpy as nm  
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


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)       




from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='poly', random_state=0)  
classifier.fit(x_train, y_train)  


    #Predicting the test set result  
y_pred= classifier.predict(x_test)  

    #Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm4= confusion_matrix(y_test, y_pred)  
print(cm4)

# Model Score
score4 = classifier.score(x_test,y_test)
print(score4)




x = [[6,2,5,3,2,7,9,2,4]]
predd = classifier.predict(x)

if predd == 2:
    Type = 'Benign' 
elif predd == 4:
    Type = 'Malignant'
print("women has class of tumor is : ", Type)




#######################  SVM ###############


##### by rbf kernel ##########





# Importing the libraries  
import numpy as nm  
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




# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)       



from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='rbf', random_state=0)  
classifier.fit(x_train, y_train)  


    #Predicting the test set result  
y_pred= classifier.predict(x_test)  

    #Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm5= confusion_matrix(y_test, y_pred)  
print(cm5)

# Model Score
score5 = classifier.score(x_test,y_test)
print(score5)





x = [[6,2,5,3,2,7,9,2,4]]
predd = classifier.predict(x)

if predd == 2:
    Type = 'Benign' 
elif predd == 4:
    Type = 'Malignant'
print("women has class of tumor is : ", Type)

