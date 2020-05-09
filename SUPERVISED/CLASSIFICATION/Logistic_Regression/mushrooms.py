
Q2. (Create a program that fulfills the following specification.)
mushrooms.csv

Import mushrooms.csv file

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one.

 

Attribute Information:

classes: edible=e, poisonous=p (outcome)

cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

cap-color: brown=n, buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

 

bruises: bruises=t, no=f

 

odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

 

gill-attachment: attached=a,descending=d,free=f,notched=n

 

gill-spacing: close=c,crowded=w,distant=d

 

gill-size: broad=b,narrow=n\

 

gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,

green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

 

stalk-shape: enlarging=e,tapering=t

 

stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

 

stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

 

stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

 

stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

 

stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

 

veil-type: partial=p,universal=u

 

veil-color: brown=n,orange=o,white=w,yellow=y

ring-number: none=n,one=o,two=t

 

ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

 

spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

 

population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

 

habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

    Perform Classification on the given dataset to predict if the mushroom is edible or poisonous w.r.t. it’s different attributes.

(you can perform on habitat, population and odor as the predictors)

    Check accuracy of the model.










import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data =  pd.read_csv("mushrooms.csv")


data.head()





features = data.iloc[:,[6,21,22]].values
labels = data.iloc[:,0].values





from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

columnTransformer = ColumnTransformer([('encoder' , OneHotEncoder(), [0])], remainder = 'passthrough')
features = np.array(columnTransformer.fit_transform(features), dtype = np.str)
features = features[:,1:]


columnTransformer = ColumnTransformer([('encoder' , OneHotEncoder(), [1])], remainder = 'passthrough')
features = np.array(columnTransformer.fit_transform(features), dtype = np.str)
features = features[:,1:]


columnTransformer = ColumnTransformer([('encoder' , OneHotEncoder(), [6])], remainder = 'passthrough')
features = np.array(columnTransformer.fit_transform(features), dtype = np.str)
features = features[:,1:]



features = features.astype('float64') 






# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.25, random_state = 0)







# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





from sklearn.linear_model import LogisticRegression
logClassifier = LogisticRegression(random_state=0)
logClassifier.fit(X_train, y_train)

# Printing Score for the Regression Model
print (logClassifier.score(X_train,y_train))





# Predicting the Test set results
labels_pred = logClassifier.predict(X_test[0].reshape(1,-1))
print (labels_pred)







