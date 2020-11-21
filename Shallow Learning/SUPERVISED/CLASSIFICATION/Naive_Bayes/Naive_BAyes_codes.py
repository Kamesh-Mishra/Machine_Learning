"""
Bernoulli Naive Bayes : It assumes that all our features are binary such that they take only two values. Means 0s can represent “word does not occur in the document” and 1s as "word occurs in the document" .

Multinomial Naive Bayes : Its is used when we have discrete data (e.g. movie ratings ranging 1 and 5 as each rating will have certain frequency to represent). 
In text learning we have the count of each word to predict the class or label.

Gaussian Naive Bayes : Because of the assumption of the normal distribution, Gaussian Naive Bayes is used in cases when all our features are continuous.
 For example in Iris dataset features are sepal width, petal width, sepal length, petal length. 
 So its features can have different values in data set as width and length can vary. We can’t represent features in terms of their occurrences. This means data is continuous. Hence we use Gaussian Naive Bayes here
"""


# more deep details

"""
For Binomial and Multinomial, let’s say we’re trying to build an email spam classifier.

Binomial - Looking through the data you notice that certain kinds of spam emails will include your email handle (the part before the @ sign) somewhere in the subject line. 
You then build a feature that captures this as 0 if it’s not present and 1 if it is.
 The algorithm will use this concept to classify emails as spam/ham and is named “Binomial” because it assumes your features are drawn from a binomial distribution.

Multinomial - Similarly as before, we notice that the more dollar signs ($) there are in an email, the more likely that email is spam.
 We can do this for many kinds of words, say (CASH or Lottery), but instead of labeling them 0 or 1, we actually count how many times each word appears in the email. 
 This helps the model by giving it information, not just on whether the word was there, but also how many times the word appeared because we know that this is a signal to help our classifier. 
 The algorithm assumes that the features are drawn from a multinomial distribution.

For Gaussian, let’s assume we’re trying to classify whether a college student can dunk a basketball based only on their height.

Gaussian - As you may recall from any intro stats class, the distribution of heights in humans is continuous and normally distributed (the normal distribution is also called a Gaussian distribution, hence the name). 
So the algorithm will look at the height of all of the students we polled and determine where the cut-off should be to maximize the model performance (usually accuracy) to classify dunkers vs non-dunkers.
"""






####################################
######################################

# Importing the libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")
# Importing the dataset  
dataset = pd.read_csv('Social_Network_Ads.csv')  
x = dataset.iloc[:, [2, 3]].values  
y = dataset.iloc[:, 4].values  
  
# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)  
  
# Feature Scaling  
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)  








################################################
################################################

# GaussianNB  



# Fitting Naive Bayes to the Training set  
from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()  
classifier.fit(x_train, y_train)  

    # Predicting the Test set results  
y_pred = classifier.predict(x_test)  

    # Making the Confusion Matrix  
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)  
print(cm)


score = classifier.score(x_test,y_test)
print(score)

# from sklearn import metrics
# metrics.accuracy_score(y_test, y_pred)




    # Visualising the Training set results  
from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
X1, X2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                     nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  
             alpha = 0.75, cmap = ListedColormap(('purple', 'green')))  
mtp.xlim(X1.min(), X1.max())  
mtp.ylim(X2.min(), X2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Naive Bayes (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()



# Visualising the Test set results  
from matplotlib.colors import ListedColormap  
x_set, y_set = x_test, y_test  
X1, X2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                     nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  
             alpha = 0.75, cmap = ListedColormap(('purple', 'green')))  
mtp.xlim(X1.min(), X1.max())  
mtp.ylim(X2.min(), X2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Naive Bayes (test set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()






###################################################
###################################################




