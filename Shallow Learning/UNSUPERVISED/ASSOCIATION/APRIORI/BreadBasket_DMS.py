
"""
Code Challenge:
dataset: BreadBasket_DMS.csv

Q1. In this code challenge, you are given a dataset which has data 
and time wise transaction on a bakery retail store.
1. Draw the pie chart of top 15 selling items.
2. Find the associations of items where min support should be 0.0025, 
min_confidence=0.2, min_lift=3.
3. Out of given results sets, show only names of the associated 
item from given result row wise.

"""



import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("E:/Machine_Learning/UNSUPERVISED/Data_files")
# Data Preprocessing
dataset = pd.read_csv('BreadBasket_DMS.csv')


dataset= dataset.iloc[:,[-2,-1]]


##########################################
##########################################

# 1. Draw the pie chart of top 15 selling items.

chart = dataset['Item'].value_counts()
chart = chart.drop(labels = 'NONE')


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = chart.index[:15]
sizes = chart[:15]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()




#########################################################
########################################################

# 2. Find the associations of items where min support should be 0.0025, 
# min_confidence=0.2, min_lift=3.

# 3. Out of given results sets, show only names of the associated 
# item from given result row wise.




tr_min= dataset['Transaction'].min()
tr_max= dataset['Transaction'].max()

transactions = []
for i in range(tr_min, tr_max):
    lis= dataset[dataset['Transaction']== i]['Item']
    transactions.append(list(lis) )


# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.0025, min_confidence = 0.2, min_lift = 3)

# Visualising the results
results = list(rules)



for item in results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
        