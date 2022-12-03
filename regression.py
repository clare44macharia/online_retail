# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:08:13 2022

@author: kasey
"""
"""""""""""""""""""""""""""""""""
Import Libraries and Definitions
"""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import seaborn as sns

#initialize label encoder
label_encoder = preprocessing.LabelEncoder()

#function to plot feature importance
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

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
#Create Profit field for predictions
df['Profit'] = df['Quantity'] * df['UnitPrice']

#Remove negative values from Quantity feature (removes ~2% of the data)
df = df[df.Quantity >= 0]
#Create copy of dataframe for results
final_df = df.copy()

#Define the features to be used in the model
cols_to_keep = ['UnitPrice', 'Country', 'InvoiceMonth', 'Description', 'Profit']
#Remove unneccessary features
cluster_df = df[cols_to_keep].copy()

#Label encode the Country and Description fields to be able to use the non-numeric field
cluster_df['Country'] = label_encoder.fit_transform(cluster_df['Country'])
cluster_df['Description'] = label_encoder.fit_transform(cluster_df['Description'])

#Save labels and features for feature importance
label = np.array(cluster_df['Profit'])
features = cluster_df.drop('Profit', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

"""""""""""""""""""""""""""""""""
Model Creation
"""""""""""""""""""""""""""""""""
#Create X and y
X,y = cluster_df.iloc[:,:-1],cluster_df.iloc[:,-1]

#Standardize the dataframe before clsutering
X_scaled = StandardScaler().fit_transform(X)

#Create training and testing groups, 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

#Create Randfom Forest regression model
regressor = RandomForestRegressor(n_estimators = 100, random_state=0)
regressor.fit(X,y)

#Predict test values, and generate predictions for all rows
y_pred = regressor.predict(X_test)
all_predictions = regressor.predict(features)

"""""""""""""""""""""""""""""""""
Model Output
"""""""""""""""""""""""""""""""""
#Generate Feature Importance Visualization
plot_feature_importance(regressor.feature_importances_,X_train.columns, 'RANDOM FOREST')

#Print Mean Absolute Error
print('Mean Absolute Error: ' ,metrics.mean_absolute_error(y_test, y_pred))

#Create field to hold predictions for all records instead of just the test set
final_df['Predicted Profit'] = all_predictions
#Create Fields for determining the actual error from all records
#Take the absolute value of the difference to determine the actual error
final_df['Difference'] = final_df['Predicted Profit'] - final_df['Profit']
final_df['abs(Difference)'] = final_df['Difference'].abs()

#Print the Mean and Standard Deviation of the entire population of results.
print("MEAN: ", final_df['abs(Difference)'].mean())
print("STD DEV: ", final_df['abs(Difference)'].std())

final_df.to_excel("Online Retail Data with RF Predictions.xlsx", index=False)

