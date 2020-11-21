

"""


Q2. This famous classification dataset first time used in Fisher’s classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems. Iris dataset is having 4 features of iris flower and one target class.

The 4 features are

SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
The target class

The flower species type is the target class and it having 3 types

Setosa
Versicolor
Virginica
The idea of implementing svm classifier in Python is to use the iris features to train an svm classifier and use the trained svm model to predict the Iris species type. To begin with let’s try to load the Iris dataset.
"""




# Importing the libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  


from sklearn import datasets 

iris = datasets.load_iris()
x = iris.data[:, :5] 
y = iris.target



# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  


#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)       




######################################
######################################

# by rbf kernel




from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='rbf', random_state=0)  
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



################################
##################################

#  by poly kernel



from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='poly', random_state=0)  
classifier.fit(x_train, y_train)  


    #Predicting the test set result  
y_pred= classifier.predict(x_test)  

    #Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm5 = confusion_matrix(y_test, y_pred)  
print(cm5)

# Model Score
score5 = classifier.score(x_test,y_test)
print(score5)


