1. Talk about scrapy -  data scrapy
2. Talk about Jupyter Notebooks
3.  Extract, Transform, Load (ETL)  - preprocessing part
4. ETL comes from Data Warehousing and stands for Extract-Transform-Load. .... sorting, joining, merging, aggregation and other operations ready to use.

5. Cognitive Healthcare
//IBM Watson
// Microsoft Cortana Intelligent Suite
// https://www.forbes.com/sites/janakirammsv/2017/01/03/how-ibm-and-microsoft-are-disrupting-the-healthcare-industry-with-cognitive-computing/#5ff942871a92
//Through the application of natural language processing (NLP), data mining, and advanced text analytics, cognitive systems can assist doctors in diagnosing and faster decision making. They optimize patient selection for clinical trials through intelligent matching. In the oncology division, these systems assist in the creation of individualized treatment plans that enhance patient and experience.

6. How IoT and DS can help learn right way to play and shot
7.  Why Python for Data Sceince - show the image from kdnuggets (http://www.kdnuggets.com/).
//Image - Why_Python_For_DataSceince.png

// Economics Times Article
++ Engineering students from MIT (Maharashtra Inst. Technology, Pune) made a robot Chintu to accompany elder people and help them. Can be useful for lonely parents.
++ As updating the course curriculum in colleges is difficult, Intel has decided to train 15000 engineers in subjects around ML.
++ Indian IT companies will need 3L data scientists by 2018.

defination AI - creating machines that perform tasks that humans do. AI can be achieved with ML or without ML.

definaton ML - training an algo to learn without being explicitly programmed. Involves feeding huge amount of data into algo and allowing it process and learn.

Deep learning - branch of ML that tries to mimic the structure of human brain..using neural network.

//Usage
// Ixigo App - Predicting which day airline will charge lower ticket price. Most airlines follow sinosudal graph for pricing tickets.
// Ixigo App also predicts about where your train ticket will confirm from waiting status.

// Algorithim trading - 
// Using big data and AI straties can you predict the GDP growth data.

// Shane and Falguni (designers) using ML to predicts which fashion and color would be hot in coming season using IBM watson.

// ML replacing radiologits job
// Wipro says it got productivity of 12000 people out of 1800 chat bots (programs that perform automated tasks)
// chat bots are not ML based...but predict the trends what may come in future



// India is home to third largest cluster of AI startups - aruond 300
// Recently Apple aquired tuplejump and Indian startup working in AI
// Google acquired Halli Labs working on AI

-------------------------------------------------
Day1
//Today we talk about classifcation machine learning model (linear classifier)
// And the first algo, we discuss is logistic regression which is part of simple linear regresson.
// Let's understand what is logistic regresson/
// Predicts the binary outcome ( YES/NO) or sometimes (MAY BE)
// Explain the data (social network.csv) -  profiles of the users
// This social network has few clients - which can put ads on this platform
// data tells who bought CAR (YES/NO) - based on Age and Salary
// How to draw the value of dependent variable out of logistic regression?
// If probability < .5, we say dependent variable is ZERO (NO)
// If probability > .5, we say dependent variable is ONE (YES)
// Show the image (logistic_regresson.jpg)
//Sample code
// Read the data
//SPlit it
// No need to labeling as not categorical data
// But need to perform the feature scaling as we need accurate predictions
// Apply the model on train data
// predict for test data
// Explain the confusion matrix (used to evaluate the performance of the model)? confusion matrix is also called as error matrix.
// in unsupervised learning it (confusion matrix) is usually called a matching matrix
// Reference - http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/// Open the above link and explain few terms out of this page

+ True positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.

+ True negatives (TN): We predicted no, and they don't have the disease.

+ False positives (FP): We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")

+ False negatives (FN): We predicted no, but they actually do have the disease. (Also known as a "Type II error.")

//Explain the graph generation correctly


// Visualization -  Prediction Boundary 
//Look at the results using graph


// Next we will build the non linear classifier


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:39:40 2018

@author: Kunal
"""

# Logistic Regression ( Classification)

# https://www.javatpoint.com/logistic-regression-in-machine-learning

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
features = dataset.iloc[:, 2:4].values
labels = dataset.iloc[:, [4]].values

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

# Predicting the Test set results
labels_pred = classifier.predict(features_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
features_set, labels_set = features_train, labels_train
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visulaizing the test set result  


# Visualising the Test set results
from matplotlib.colors import ListedColormap
features_set, labels_set = features_test, labels_test
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
        c = ListedColormap(('purple', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#understand meshgrid
#https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python
#https://www.kaggle.com/arthurtok/decision-boundaries-visualised-via-python-plotly
#IMportant
#https://stackoverflow.com/questions/20045994/how-do-i-plot-the-decision-boundary-of-a-regression-using-matplotlib



# evenly sampled points
x_min, x_max = features_train[:, 0].min() - .5, features_train[:, 0].max() + .5
y_min, y_max = features_train[:, 1].min() - .5, features_train[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

#plot background colors
ax = plt.gca()
Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=.5)
cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=.5)
plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)

# Plot the points
ax.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 1')
ax.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 2')

# make legend
plt.legend(loc='upper left')


#version 4 for decision boundary
#http://yooooh.net/2016/02/11/plot-classification-decision-boundary/

# Step size of the mesh. Decrease to increase the quality of the VQ.
#h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].    

# Plot the decision boundary. For that, we will assign a color to each
   
"""
The purpose of meshgrid function in Python is to create a rectangular grid out of an array of x values and an array of y values. Suppose you want to create a grid where you have a point at each integer value between 0 and 4 in both the x and y directions.
"""

x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Obtain labels for each point in mesh using the model.
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the points
plt.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 2')
plt.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 1')

plt.contourf(xx, yy, Z, alpha=0.3)
plt.show()



# -*- coding: utf-8 -*-
"""
The Iris flower data set or Fisher's Iris data set is a multivariate
data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper
The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.
"""
#We import this dataset already in sklearn, and perform logisitic regression based on petals and sepals properties to classify the iris species
from sklearn.datasets import load_iris

iris = load_iris() # Loading Iris Dataset
print (iris.feature_names)  # Column Names for the iris dataset

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logClassifier = LogisticRegression(random_state=0)
logClassifier.fit(features_train, labels_train)

# Printing Score for the Regression Model
print (logClassifier.score(features_train,labels_train))
# Predicting the Test set results
labels_pred = logClassifier.predict(features_test[0].reshape(1,-1))
print (labels_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)


"""
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
"""
# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/files")
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


"""
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""

#Visulaizing the test set result  
from matplotlib.colors import ListedColormap  
x_set, y_set = X_test, y_test  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
             alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.title('Logistic Regression (Test set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  



"""


Q1. (Create a program that fulfills the following specification.)
affairs.csv


Import the affairs.csv file.

It was derived from a survey of women in 1974 by Redbook magazine, in which married women were asked about their participation in extramarital affairs.

Description of Variables

The dataset contains 6366 observations of 10 variables:(modified and cleaned)

rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)

age: women's age

yrs_married: number of years married

children: number of children

religious: women's rating of how religious she is (1 = not religious, 4 = strongly religious)

educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree)

occupation: women's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 = professional with advanced degree)

occupation_husb: husband's occupation (same coding as above)

affair: outcome 0/1, where 1 means a woman had at least 1 affair.

    Now, perform Classification using logistic regression and check your model accuracy using confusion matrix and also through .score() function.

NOTE: Perform OneHotEncoding for occupation and occupation_husb, since they should be treated as categorical variables. Careful from dummy variable trap for both!!

    What percentage of total women actually had an affair?

(note that Increases in marriage rating and religiousness correspond to a decrease in the likelihood of having an affair.)

    Predict the probability of an affair for a random woman not present in the dataset. She's a 25-year-old teacher who graduated college, has been married for 3 years, has 1 child, rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.

Optional

    Build an optimum model, observe all the coefficients.

"""




LOGISTIC REGRESSION
# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("E:/Machine_Learning/files")
# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
features = dataset.iloc[:, [2, 3]].values
labels = dataset.iloc[:, 4].values

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

# Predicting the Test set results
labels_pred = classifier.predict(features_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
features_set, labels_set = features_train, labels_train
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
features_set, labels_set = features_test, labels_test
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

CODE CHALLENGE

Q1. (Create a program that fulfills the following specification.)
affairs.csv


Import the affairs.csv file.

It was derived from a survey of women in 1974 by Redbook magazine, in which married women were asked about their participation in extramarital affairs.

Description of Variables

The dataset contains 6366 observations of 10 variables:(modified and cleaned)

rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)

age: women's age

yrs_married: number of years married

children: number of children

religious: women's rating of how religious she is (1 = not religious, 4 = strongly religious)

educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree)

occupation: women's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 = professional with advanced degree)

occupation_husb: husband's occupation (same coding as above)

affair: outcome 0/1, where 1 means a woman had at least 1 affair.

    Now, perform Classification using logistic regression and check your model accuracy using confusion matrix and also through .score() function.

NOTE: Perform OneHotEncoding for occupation and occupation_husb, since they should be treated as categorical variables. Careful from dummy variable trap for both!!

    What percentage of total women actually had an affair?

(note that Increases in marriage rating and religiousness correspond to a decrease in the likelihood of having an affair.)

    Predict the probability of an affair for a random woman not present in the dataset. She's a 25-year-old teacher who graduated college, has been married for 3 years, has 1 child, rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.

Optional

    Build an optimum model, observe all the coefficients.
