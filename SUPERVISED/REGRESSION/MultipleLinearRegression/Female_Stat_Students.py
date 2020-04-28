


"""


Q2. (Create a program that fulfills the following specification.)
Female_Stats.Csv

Female Stat Students

 

Import The Female_Stats.Csv File

The Data Are From N = 214 Females In Statistics Classes At The University Of California At Davi.

Column1 = Student’s Self-Reported Height,

Column2 = Student’s Guess At Her Mother’s Height, And

Column 3 = Student’s Guess At Her Father’s Height. All Heights Are In Inches.

 

    Build A Predictive Model And Conclude If Both Predictors (Independent Variables) Are Significant For A Students’ Height Or Not
    When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.
    When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.


"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
import os

os.chdir("E:/Machine_Learning/files")

# Importing the dataset
dataset = pd.read_csv('Female_Stats.csv')


#temp = dataset.values
labels = dataset.iloc[:,[0]].values
features = dataset.iloc[:, [1,2]].values







# Building the optimal model using Backward Elimination
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm

#This is done because statsmodels library requires it to be done for constants.
features = np.append(arr = np.ones((214, 1)), values = features, axis = 1)




features_opt = features[:, [0,1,2]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()




list_ = list(range(len(features[0])))

while(True):
    regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()

    if regressor_OLS.pvalues.max()>=0.05:
        del list_[regressor_OLS.pvalues.argmax()]
        print(list_)
        features_opt = features[:, list_]
    else:
        print("Both Predictors (Independent Variables) Are Significant For A Students’ Height")
        break
    

# regressor_OLS.pvalues




 # When Father’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Mother’s Height.

features= dataset.iloc[:,[1]].values       

labels= dataset.iloc[:,[0]].values       
    

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

pred= regressor.predict(features_test)

print(pd.DataFrame(zip(pred, labels_test)))




score = regressor.score(features_train, labels_train )
print(score)

score = regressor.score(features_test, labels_test)
print(score)




 # When Mother’s Height Is Held Constant, The Average Student Height Increases By How Many Inches For Each One-Inch Increase In Father’s Height.



features= dataset.iloc[:,[2]].values       

labels= dataset.iloc[:,[0]].values       
    

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

pred= regressor.predict(features_test)

print(pd.DataFrame(zip(pred, labels_test)))




score1 = regressor.score(features_train, labels_train )
print(score1)

score1 = regressor.score(features_test, labels_test)
print(score1)



