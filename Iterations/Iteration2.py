#How to run:
# python3 Clean-Copy1.py
# make sure katkam-scaled folder in directory and yvr-weather directory present

#Iteration 1:
# -read images as gray scale
# -normalize divde by 255
# -clean the weather description so that "Rain Showers" and "Rain" is considered the same for example
# -CLEANED: Mostly, Mainly, Rain Showers, Heavy Rain
# -TODO/TRY: Clean up Drizzle?
# -use only the non-null weather column for training and testing
# -use k-nearest neighbors

#from sklearn.neighbors import KNeighborsClassifier #0.689
#from sklearn.multiclass import OutputCodeClassifier
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC #--C=1e5 0.6768
#from sklearn.naive_bayes import GaussianNB #0.646341463415
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.misc import imread
import pandas as pd
import numpy as np
import glob
import re

import pickle

#save trained model to a persistent format
def pickleModel(model):
  with open("weatherml.pkl", "wb") as fp:
      pickle.dump(model, fp, 2)

  with open("image", "wb") as fp_train:
    img_format = np.zeros((1,49152), dtype=np.float32)
    pickle.dump(img_format, fp_train)

  fp.close()
  fp_train.close()

  #img_dict = dict(zip(train_set.shape[1],range(train_shape[1])))
  #pickel.dump(img_dict, fp_train)


#Reads all image files from /katkam-scaled folder
#Converts the images to gray scale and concatentates the timestamp
#Timestamp taken from the name of the original image file
def ReadImage2GrayScale(folderpath, rgb=True):
  row = 0
  image_data = pd.DataFrame(columns=("image", "timestamp"))

  if (rgb):

    for path in sorted(glob.glob(folderpath + "/*")):
      matches = re.search("\d+", path)
      image_data.loc[row] = [imread(path, flatten=True).ravel(), matches.group()]
      row += 1
  else:

    for path in sorted(glob.glob(folderpath + "/*")):
      matches = re.search("\d+", path)
      image_data.loc[row] = [imread(path).ravel(), matches.group()]
      row += 1

  image_data["timestamp"] = pd.to_datetime(image_data["timestamp"])
  return image_data


#Load the weather data & rename the time column to timestamp
def CleanWeatherData(weather_data):
  # data quality <-- look at later?
  weather_data.drop(["Year","Month","Day","Time","Temp Flag", "Dew Point Temp Flag",
                     "Rel Hum Flag", "Wind Dir Flag", "Wind Spd Flag",
                     "Visibility Flag", "Stn Press Flag",
                     "Hmdx Flag", "Wind Chill Flag", "Hmdx", "Wind Chill",
                     "Data Quality"], axis=1, inplace=True)
  weather_data.rename(columns={'Date/Time':'timestamp'}, inplace=True)
  weather_data = weather_data[~weather_data["Weather"].isnull()]


# Clean up words descriptions Mostly, Mainly, Rain Showers, Heavy Rain, Drizzle, Thunderstorm, Freezing
# Only categories allowed: Clear, Cloudy, Fog, Rain, combinations...
def CleanDescription(weather):

  regex_string = "(Freezing[\s]+)*(Heavy[\s]+)*(Moderate[\s]+)*(Mostly[\s]+)*(Mainly[\s]+)*([\s]+Showers)*([\s]+Pellets)*"
  weather["Weather"] = weather["Weather"].str.replace(regex_string, "")

  rain_snow = "Rain,Snow"
  weather["Weather"] = weather["Weather"].str.replace(rain_snow, "Rain")

  no_thunder = "Thunderstorms"
  weather["Weather"] = weather["Weather"].str.replace(no_thunder, "Cloudy")

  no_drizzle = "Drizzle"
  weather["Weather"] = weather["Weather"].str.replace(no_drizzle, "Cloudy")
  #weather["Weather"].replace("", np.nan, inplace=True)

  fog = "Cloudy,Fog"
  weather["Weather"] = weather["Weather"].str.replace(fog, "Fog")

  snow_fog = "Snow,Fog"
  weather["Weather"] = weather["Weather"].str.replace(snow_fog, "Fog")

  rain_fog = "Rain,Fog"
  weather["Weather"] = weather["Weather"].str.replace(rain_fog, "Fog")



#Takes the original weather data loaded as Dataframe and converts to ML friendly format
def dataFrame2Array(images, train_set):
  # reshape to (_,49152) (192x256 image to 1-d)

  for i in range(train_set.shape[0]):
      images[i] = train_set.image.iloc[i] / 255



def matchAndSortData(image_data, weather_data):

  #clean up weather description and drop unnecessary columns
  CleanWeatherData(weather_data)
  CleanDescription(weather_data)

  weather_times = list(weather_data.timestamp.map(lambda x: x))
  image_data = image_data[image_data.timestamp.isin(weather_times)]

  image_times = list(image_data.timestamp.map(lambda x: x))
  weather_data = weather_data[weather_data.timestamp.isin(image_times)]

  image_data = image_data.sort_values(by='timestamp')
  weather_data = weather_data.sort_values(by='timestamp')

  return image_data, weather_data


# take all yvr weather files and merge them into one dataframe
def readMergeData(drop_null=True):
  files = [f for f in sorted(glob.glob("./yvr-weather/*"))]

  data_set = pd.read_csv(files[0], sep=',', skiprows=16, parse_dates=[0])
  for i in range(1,13):
    weather_data = pd.read_csv(files[i], sep=',', skiprows=16, parse_dates=[0])
    data_set = data_set.append(weather_data)

  data_set = data_set[~data_set["Weather"].isnull()]

  return data_set


def main():

  print("Reading image data...")
  image_data = ReadImage2GrayScale("./katkam-eq", rgb=False)
  # get list of weather data csv files

  weather_data = readMergeData()

  print("Cleaning weather data...")
  print("Spliting data...")

  image_data, weather_data = matchAndSortData(image_data, weather_data)
  X_train_img, X_test_img, y_train, y_test = train_test_split(image_data, weather_data, train_size=0.9)

  X_train = np.zeros((X_train_img.shape[0],49152), dtype=np.float32)
  X_test  = np.zeros((X_test_img.shape[0],49152), dtype=np.float32)

  dataFrame2Array(X_train, X_train_img)
  dataFrame2Array(X_test, X_test_img)

  print("Fitting data...")
  #model = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree' ,leaf_size=20)
  #model = OutputCodeClassifier(LinearSVC(random_state=2), code_size=2, random_state=3)
  #model = SVC(C=1e5)
  #model = GaussianNB()
  model = RandomForestClassifier(n_estimators=50)


  model.fit(X_train, y_train.Weather)
  #print(model.score(X_test, y_test.Weather))
  y_predicted = model.predict(X_test)
  print(classification_report(y_test.Weather,y_predicted))


  pickleModel(model)



if __name__ == '__main__':
  main()
