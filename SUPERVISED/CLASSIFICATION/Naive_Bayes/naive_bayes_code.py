

#   https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

#  https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/

#  https://www.javatpoint.com/machine-learning-naive-bayes-classifier









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("E:/Machine_Learning/SUPERVISED/Data_files")
dataset = pd.read_csv("Match_Making.csv")
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)


##########################################################


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(features, labels)
# print(model)


labels_pred = model.predict(features_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

score = model.score(features_test,labels_test)
print(score)


from sklearn import metrics
metrics.accuracy_score(labels_test, labels_pred)
###################################################


from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(features,labels)

labels_pred = model.predict(features_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(labels_test, labels_pred)
print(cm1)

score1 = model.score(features_test,labels_test)
print(score1)


from sklearn import metrics
metrics.accuracy_score(labels_test, labels_pred)
###########################################################


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(features, labels)

labels_pred = model.predict(features_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(labels_test, labels_pred)
print(cm2)

score2 = model.score(features_test,labels_test)
print(score2)



from sklearn import metrics
metrics.accuracy_score(labels_test, labels_pred)

################################################################

# comaparison overall GsnNB   BernNB    multinmNB




print(cm)
print(score)

print(cm1)
print(score1)

print(cm2)
print(score2)
