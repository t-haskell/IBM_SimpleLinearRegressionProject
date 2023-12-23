'''
Project by -> Thomas Haskell

Topic -> Multiple Linear Regression
Source -> IBM Machine Learning with Python Certification Course

Objectives:
1. Learn how to implement multiple linear regression with scikit-learn
2. Learn how create, train, and test this model
'''

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

## Downloading data set from IBM Cloud
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
path= "/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
import urllib.request
import ssl

global IND_VAR
IND_VAR = "FUELCONSUMPTION_COMB"

# workaround for SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def download(url, path, filename):
    urllib.request.urlretrieve(url, filename)
    print("Download Complete")
    
# reading data in
download(url, path, "FuelConsumptionCo2.csv")
df = pd.read_csv("FuelConsumptionCo2.csv")
print(df.head())

# potential features for regression
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# visualize engine size and CO2 emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

####################  Model Training  ##################

# splitting traininng and testing datasets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Optimizing coefficients for line of best fit (Training)
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

####################  Model Testing/Evaluation  ##################

# For multiple linear regression, a good method to predict is to use Ordinary Least Squares (OLS),
# which is a linear regression model that minimizes the sum of squared errors between the dependent
# variable and the independent variable.
