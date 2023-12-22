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

    """ connection = http.client.HTTPSConnection(url)
    # using GET method to find the file in cloud
    connection.request("GET", path)
    response = connection.getresponse()
    if response.status == 200:
        with open(filename, "wb") as f:
            # saving file to local disk
            f.write(response.read())
            print(f"Download of '{filename}' is Successful")
    else:
        print(f"Download of '{filename}' is Failed")
    connection.close() """
    
# now calling the function to grab the data
download(url, path, "FuelConsumptionCo2.csv")
#path = "FuelConsumptionCo2.csv"
    
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

###########  Model Training and Testing ###########








            
    



