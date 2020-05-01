
import matplotlib.pyplot as plt
import numpy as np

HOURS = [0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.00,5.50]
PASS = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]

plt.scatter(HOURS, PASS)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(np.array(HOURS).reshape(-1,1), np.array(PASS).reshape(-1,1)) 


#Visualize the best fit line
import matplotlib.pyplot as plt

# Visualising the  results
plt.scatter(HOURS, PASS, color = 'red')
plt.plot(HOURS, regressor.predict(np.array(HOURS).reshape(-1,1)), color = 'blue')
plt.title('Study Hours and Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score: Marks')
plt.show()












import pandas as pd
import os

"""
We will look at data regarding coronary heart disease (CHD) in South Africa. 
The goal is to use different variables such as tobacco usage, family history, 
ldl cholesterol levels, alcohol usage, obesity and more.
"""

os.chdir("E:/Machine_Learning/files")
heart = pd.read_csv('Heart_Disease.csv')  
heart.head()

labels = heart.iloc[:,9].values 
features = heart.iloc[:,:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)

pd.DataFrame(labels_pred, labels_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

#http://vassarstats.net/logreg1.html
