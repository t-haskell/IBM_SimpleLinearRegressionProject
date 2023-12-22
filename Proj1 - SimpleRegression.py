''' 
Project by -> Thomas Haskell

Topic -> Simple Linear Regression
Source -> IBM Machine Learning with Python Certification Course

Below is a translation of a Jupyter notebook program to Python. Using IBM technologies, 
this program is able to predict the CO2 emissions based on varying features of a car.

The focus of this project is on simple linear regression modeling, where one independent
variable is used to predict the dependent variable (CO2 Emissions).

'''


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Downloading data set from IBM Cloud
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
path= "/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
import http.client
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download(url, path, filename):
    urllib.request.urlretrieve(url, filename)
    print("Download Complete")
    
# now calling the function to grab the data
download(url, path, "FuelConsumptionCo2.csv")
    
# putting the data into a pandas dataframe
df = pd.read_csv("FuelConsumptionCo2.csv")
print(df.head())

# summarize the data
print(df.describe())

# selecting features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))

# visualize the features
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()

# Combined Fuel Consumption vs CO2 Emission
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# Engine Size vs CO2 Emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Number of Cylinders vs CO2 Emission
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="red")
plt.xlabel("# of Cylinders")
plt.ylabel("Emissions")
plt.show()

###########  Model Training  ###########

# splitting traininng and testing datasets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# visualizing the training data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# visualizing the testing data distribution
plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#using sklearn to model the data
from sklearn import linear_model

regrML = linear_model.LinearRegression()
train_x = np.asanarray(train[['ENGINESIZE']])
train_y = np.asanarray(train[['CO2EMISSIONS']])
regrML.fit(train_x, train_y)
# printing the found optimal coefficients from line above
print("Coefficients: ", regrML.coef_)
print("Intercept: ", regrML.intercept_)

# plotting line of regression
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train.ENGINESIZE, regrML.coef_[0][0]*train.ENGINESIZE + regrML.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


####################  Model Testing/Evaluation  ##################

# focusing on Mean Squared Error (MSE)
from sklearn.metrics import r2_score

# Calculating predictions using test data
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
predictions = regrML.predict(test_x)    # is vector of dependent variable predictions

mae = np.mean(np.abs(predictions - test_y))
mse = np.mean((predictions - test_y) ** 2)
r2 = r2_score(test_y, predictions)

print(f"Mean Squared Error ('Residual sum of squares'): {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")







            
    



