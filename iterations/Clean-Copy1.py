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
from scipy.misc import imread
import pandas as pd
import numpy as np
import glob
import re

import pickle

#save trained model to a persistent format
def pickelModel(model):
    with open("weatherml.pkl", "wb") as fp:
        pickle.dump(model, fp, 2)

    #fp_train = open("image", "wb")
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
        image_data.loc[row] = [imread(path).ravel(), matches.group()]
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


# Clean up words descriptions Mostly, Mainly, Rain Showers, Heavy Rain, Drizzle, Thunderstorm, Freezing
# Only categories allowed: Clear, Cloudy, Fog, Rain, combinations...
def CleanDescription(weather):
    #re.sub("(Moderate[\s]+)*(Mostly[\s]+)*(Mainly[\s]+)*([\s]+Showers)*","", weather["Weather"])
    regex_string = "(Freezing[\s]+)*(Heavy[\s]+)*(Moderate[\s]+)*(Mostly[\s]+)*(Mainly[\s]+)*([\s]+Showers)*([\s]+Pellets)*"
    weather["Weather"] = weather["Weather"].str.replace(regex_string, "")

    # no_drizzle = "(,Drizzle)?(Drizzle,)?(Drizzle)?"
    # weather["Weather"] = weather["Weather"].str.replace(no_drizzle, "")
    # weather["Weather"].replace("", np.nan, inplace=True)

    # rain_snow = "Rain,Snow"
    # weather["Weather"] = weather["Weather"].str.replace(rain_snow, "Rain")

    # no_thunder = "Thunderstorms"
    # weather["Weather"] = weather["Weather"].str.replace(no_thunder, "Cloudy")



#Takes the original weather data loaded as Dataframe and converts to ML friendly format
def dataFrame2Array(images, train_set):
    # reshape to (_,49152) (192x256 image to 1-d)

    for i in range(train_set.shape[0]):
        images[i] = train_set.image.iloc[i] / 255



#matches data for training and testing predictions...
#must use two different weather files and set them to train and
#test data respectively
def splitDataWithNanValues(images, weather, impute=False):
    
    if( impute ):
      """
      Impute to account for difference in locations which the image
      data and weather was collected (images of English Bay, weather data from YVR)

      Fills in data from most recent non-blank value
      Good probability that weather from previous hour is the same
      """

      weather["Weather"].fillna(method='ffill', inplace=True)
      train = weather

    else:
      train = weather[~weather["Weather"].isnull()]

    train_times = list(train.timestamp.map(lambda x: x))

    img_train = images[images.timestamp.isin(train_times)]
    train_img_times = list(img_train.timestamp.map(lambda x: x))

    y_train = train[train.timestamp.isin(train_img_times)]
    X_train = np.zeros((img_train.shape[0],49152), dtype=np.float32)

    dataFrame2Array(X_train, img_train)

    return X_train, y_train


def readMergeData():
  files = [f for f in sorted(glob.glob("./yvr-weather/*"))]

  data_set = pd.read_csv(files[0], sep=',', skiprows=16, parse_dates=[0])
  for i in range(1,10):
    weather_data = pd.read_csv(files[i], sep=',', skiprows=16, parse_dates=[0])
    data_set = data_set.append(weather_data)
    #del(weather_data)


  test_set = pd.read_csv(files[10], sep=',', skiprows=16, parse_dates=[0])
  for j in range(11,12):
    weather_data = pd.read_csv(files[i], sep=',', skiprows=16, parse_dates=[0])
    test_set = test_set.append(weather_data)
    #del(weather_data)


  return data_set, test_set

def main():

  print("Reading image data...")
  image_data = ReadImage2GrayScale("./katkam-eq", rgb=False)
  # get list of weather data csv files
  
  weather_data, test_set = readMergeData()

  print("Cleaning weather data...")

  CleanDescription(weather_data)
  CleanDescription(test_set)

  CleanWeatherData(weather_data)
  CleanWeatherData(test_set)

  print("Spliting data...")

  X_train, y_train = splitDataWithNanValues(image_data, weather_data)
  X_test, y_test = splitDataWithNanValues(image_data, test_set)

  print("Train size: " + str(X_train.shape[0]) + " y: " + str(y_train.shape[0]))
  print("Test size: " + str(X_test.shape[0]) + " y: " + str(y_test.shape[0]))


  #model = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree' ,leaf_size=20)
  #model = OutputCodeClassifier(LinearSVC(random_state=2), code_size=2, random_state=3)
  #model = SVC(C=1e5)
  #model = GaussianNB()
  model = RandomForestClassifier(n_estimators=50)


  model.fit(X_train, y_train.Weather)
  #print(model.score(X_test, y_test.Weather))
  y_predicted = model.predict(X_test)
  print(classification_report(y_test.Weather,y_predicted))


  #pickelModel(model)



if __name__ == '__main__':
  main()
