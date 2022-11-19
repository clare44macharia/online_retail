# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:31:01 2022

@author: kasey
"""

"""""""""""""""""""""""""""""""""
Import Libraries and Definitions
"""""""""""""""""""""""""""""""""
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#initialize label encoder
label_encoder = preprocessing.LabelEncoder()

#save list of countries for encoding
list_of_countries = ['Australia', 'Austria', 'Bahrain', 'Belgium', 'Brazil', 'Canada',
       'Channel Islands', 'Cyprus', 'Czech Republic', 'Denmark', 'EIRE',
       'European Community', 'Finland', 'France', 'Germany', 'Greece',
       'Hong Kong', 'Iceland', 'Israel', 'Italy', 'Japan', 'Lebanon',
       'Lithuania', 'Malta', 'Netherlands', 'Norway', 'Poland',
       'Portugal', 'RSA', 'Saudi Arabia', 'Singapore', 'Spain', 'Sweden',
       'Switzerland', 'USA', 'United Arab Emirates', 'United Kingdom',
       'Unspecified']

"""""""""""""""""""""""""""""""""
Import Data
"""""""""""""""""""""""""""""""""
# Read in excel file, changing sheet name when necessary 
df = pd.read_csv(r'C:\Users\kasey\Documents\Online_Retail.csv', encoding = 'unicode_escape')

"""""""""""""""""""""""""""""""""
Data Cleaning
"""""""""""""""""""""""""""""""""
#Change InvoiceDate to InvoiceMonth
df['InvoiceMonth'] = pd.DatetimeIndex(df['InvoiceDate']).month

#Remove negative values from Quantity feature (removes ~2% of the data)
df = df[df.Quantity >= 0]
#Create copy of dataframe for results
final_df = df.copy()

#Define the features to be used in the model
cols_to_keep = ['Quantity', 'UnitPrice', 'Country', 'InvoiceMonth']
#Remove unneccessary features
cluster_df = df[cols_to_keep].copy()

#Label encode the Country field to be able to use the non-numeric field
cluster_df['Country'] = label_encoder.fit_transform(cluster_df['Country'])

"""""""""""""""""""""""""""""""""
Model Creation
"""""""""""""""""""""""""""""""""
#Standardize the dataframe before clsutering
X = StandardScaler().fit_transform(cluster_df)

# =============================================================================
# # Elbow Method for K means
# # Run Once, then comment out to prevent long run time
# # Import ElbowVisualizer
# 
# model = KMeans()
# # k is range of number of clusters.
# visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
# visualizer.fit(X)        # Fit data to visualizer
# visualizer.show()        # Finalize and render figure
# 
# #RESULT = 8 clusters ideal
# =============================================================================

#Create model, generate and label clusters
kmeans_model = KMeans(n_clusters=8, random_state=0).fit(X)

#Create field in the dataframe to contain cluster label
final_df['Cluster'] = kmeans_model.labels_

final_df.to_excel("Online Data with Clusters.xlsx", index=False)

        




