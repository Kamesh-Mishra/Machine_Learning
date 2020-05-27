"""

Q1. (Create a program that fulfills the following specification.)
iq_size.csv

Are a person's brain size and body size (Height and weight) predictive of his or her intelligence?

 

Import the iq_size.csv file

It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students

Column 2:  The brain size (MRI) of students (given as count/10,000).

Column 3: The height (Height) of students (inches)

Column 4: The weight (Weight) of student (pounds)

    What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
    Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or brain size.


"""



# Importing the libraries
import numpy as np
import pandas as pd
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")
# Importing the dataset
dataset = pd.read_csv('iq_size.csv')


features = dataset.iloc[:, 1:].values
labels = dataset.iloc[:, 0].values




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)


# Predicting the Test set results
Pred = regressor.predict(features_test)

print (pd.DataFrame(zip(Pred, labels_test)))



x = [90,70,150]
x= np.array(x)
x = x.reshape(1,-1)

print(regressor.predict(x))






# Building the optimal model using Backward Elimination

#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm

#This is done because statsmodels library requires it to be done for constants.
features = np.append(arr = np.ones((38, 1)), values = features, axis = 1)




features_opt = features[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()




regressor_OLS.pvalues


"""

#################### manual process of backward elimination  #####################

features_opt = features[:, [0, 1, 2]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()



regressor_OLS.pvalues



features_opt = features[:, [ 1, 2]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()


regressor_OLS.pvalues




features_opt = features[:, [1]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()




regressor_OLS.pvalues

"""


############# loop process of backward elimination ####################


list_ = list(range(len(features[0])))

while(True):
    regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()

    if regressor_OLS.pvalues.max()>=0.05:
        del list_[regressor_OLS.pvalues.argmax()]
        print(list_)
        features_opt = features[:, list_]
    else:
        break
    
    
print("Brain is most useful in predicting intelligence")
