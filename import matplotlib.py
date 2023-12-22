import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Downloading data set from IBM Cloud
path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
import http.client
def download(url, filename):
    connection = http.client.HTTPConnection(url)
    # using GET method to find the file in cloud
    connection.request("GET", filename)
    response = connection.getresponse()
    with open(filename, "wb") as f:
        # saving file to local disk
        f.write(response.read())
    connection.close()    
    
# now calling the function to grab the data
download(path, "FuelConsumption.csv")
path = "FuelConsumption.csv"
    
# putting the data into a pandas dataframe
df = pd.read_csv("FuelConsumption.csv")

df.head()


            
    



