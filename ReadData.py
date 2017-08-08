from scipy.misc import imread
import pickle
import pandas as pd
import numpy as np
#from zipfile import ZipFile
import glob
import re

'''-------------------------------------------------
  Persist the model with Pickle

'''

#save trained model to a persistent format
def pickleModel(model, name):
  with open(name + ".pkl", "wb") as fp:
      pickle.dump(model, fp, 2)

  fp.close()


'''-------------------------------------------------
  Read and Inital Column Dropping of YVR data below

'''

# take all yvr weather files and merge them into one dataframe
def readMergeData(debug=False):

  if debug:
    limit = 1
  else:
    limit = 13

  files = [f for f in sorted(glob.glob("./yvr-weather/*"))]
  # zip_files = ZipFile("yvr.zip")
  # files = [f for f in sorted(zip_files.namelist())]

  data_set = pd.read_csv(files[0], sep=',', skiprows=16, parse_dates=[0])
  for i in range(1,limit):
    weather_data = pd.read_csv(files[i], sep=',', skiprows=16, parse_dates=[0])
    data_set = data_set.append(weather_data)

  #data_set = data_set[~data_set["Weather"].isnull()]

  return data_set


#Load the weather data & rename the time column to timestamp
def CleanWeatherData(weather_data):
  
  weather_data.drop(["Year","Month","Day","Time","Temp Flag", "Dew Point Temp Flag",
                     "Rel Hum Flag", "Wind Dir Flag", "Wind Spd Flag",
                     "Visibility Flag", "Stn Press Flag",
                     "Hmdx Flag", "Wind Chill Flag", "Hmdx", "Wind Chill",
                     "Data Quality"], axis=1, inplace=True)
  weather_data.rename(columns={'Date/Time':'timestamp'}, inplace=True)


'''--------------------------------------------------
 Read and Format Image Data Functions Below

'''

#Timestamp taken from the name of the original image file
def ReadImage2GrayScale(folderpath, rgb=True):
  row = 0
  image_data = pd.DataFrame(columns=("image", "timestamp"))

  if (rgb):
    # with ZipFile(folderpath) as zf:
    #   for name in zf.namelist():
    #     matches = re.search("\d+", name)
    #     image_data.loc[row] = [imread(name, flatten=True).ravel(), matches.group()]
    #     row += 1

    for path in sorted(glob.glob(folderpath + "/*")):
      matches = re.search("\d+", path)
      image_data.loc[row] = [imread(path, flatten=True).ravel(), matches.group()]
      row += 1
  else:

    # with ZipFile(folderpath) as zf:
    #   for name in zf.namelist():
    #     matches = re.search("\d+", name)
    #     image_data.loc[row] = [imread(name).ravel(), matches.group()]
    #     row += 1

    for path in sorted(glob.glob(folderpath + "/*")):
      matches = re.search("\d+", path)
      image_data.loc[row] = [imread(path).ravel(), matches.group()]
      row += 1

  image_data["timestamp"] = pd.to_datetime(image_data["timestamp"])
  return image_data


#Takes the original weather data loaded as Dataframe and converts to ML friendly format
def dataFrame2Array(images, train_set):
  # reshape to (_,49152) (192x256 image to 1-d)

  for i in range(train_set.shape[0]):
      images[i] = train_set.image.iloc[i] / 255

