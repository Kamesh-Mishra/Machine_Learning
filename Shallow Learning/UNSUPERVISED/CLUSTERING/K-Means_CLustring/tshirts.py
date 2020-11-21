"""

Q2. (Create a program that fulfills the following specification.)
tshirts.csv


T-Shirt Factory:

You own a clothing factory. You know how to make a T-shirt given the height 
and weight of a customer.

You want to standardize the production on three sizes: small, medium, and large. 
How would you figure out the actual size of these 3 types of shirt to better 
fit your customers?

Import the tshirts.csv file and perform Clustering on it to make sense out of 
the data as stated above.

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("E:/Machine_Learning/UNSUPERVISED/Data_files")

dataset = pd.read_csv("tshirts.csv")

dataset.plot.scatter(x = "height (inches)", y = "weight (pounds)")


features = dataset.iloc[:,[1,2]].values

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)    

#Now plot it        
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# https://medium.com/@iSunilSV/data-science-python-k-means-clustering-eed68b490e02

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(features)

# init = k-means++     ## theory of syntax 


kmeans.cluster_centers_



plt.scatter(features[y_kmeans==0, 0], features[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(features[y_kmeans==1, 0], features[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(features[y_kmeans==2, 0], features[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')

#Plot the centroid. This time we're going to use the cluster centres  attribute that 
#returns here the coordinates of the centroid.

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label ='Centroids')
plt.title('Clusters of sizes')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()



# DataFrame of clusters

large_size_cluster = pd.DataFrame((zip(features[y_kmeans==0, 0], features[y_kmeans==0, 1])), columns = ["Height","Weight"] )
print(large_size_cluster)


medium_size_cluster = pd.DataFrame(zip(features[y_kmeans==1, 0], features[y_kmeans==1, 1]), columns = ["Height","Weight"] )
print(large_size_cluster)

small_size_cluster = pd.DataFrame(zip(features[y_kmeans==2, 0], features[y_kmeans==2, 1]), columns = ["Height","Weight"] )
print(small_size_cluster)



# visualization of each cluster

large_size_cluster.plot.scatter(x = "Height", y = "Weight", c = 'green')

medium_size_cluster.plot.scatter(x = "Height", y = "Weight", c = 'red')

small_size_cluster.plot.scatter(x = "Height", y = "Weight", c = 'black')


