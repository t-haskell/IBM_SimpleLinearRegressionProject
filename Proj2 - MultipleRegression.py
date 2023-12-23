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