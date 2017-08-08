#How to run:
# python3 Iteration3.py
# make sure katkam-scaled folder in directory and yvr-weather directory present

#Iteration 3:
# -read images as equalized gray scale
# -normalize divde images by 255
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
from optparse import OptionParser
#from zipfile import ZipFile
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ReadData import pickleModel, readMergeData, CleanWeatherData, ReadImage2GrayScale, dataFrame2Array
import pandas as pd
import numpy as np

# Clean up words descriptions Mostly, Mainly, Rain Showers, Heavy Rain, Drizzle, Thunderstorm, Freezing
# Only categories allowed: Clear, Cloudy, Fog, Rain, combinations...
def CleanDescription(weather):

  regex_string = "(Freezing[\s]+)*(Heavy[\s]+)*(Moderate[\s]+)*(Mostly[\s]+)*(Mainly[\s]+)*([\s]+Showers)*([\s]+Pellets)*"
  weather["Weather"] = weather["Weather"].str.replace(regex_string, "")

  rain_snow = "Rain,Snow"
  weather["Weather"] = weather["Weather"].str.replace(rain_snow, "Snow")

  no_thunder = "Thunderstorms"
  weather["Weather"] = weather["Weather"].str.replace(no_thunder, "Cloudy")

  no_drizzle = "Drizzle"
  weather["Weather"] = weather["Weather"].str.replace(no_drizzle, "Cloudy")
  #weather["Weather"].replace("", np.nan, inplace=True)

  #remove artifically generated weather string
  remove = "Rain,Cloudy"
  weather["Weather"] = weather["Weather"].str.replace(remove, "Rain")

  fog = "Cloudy,Fog"
  weather["Weather"] = weather["Weather"].str.replace(fog, "Fog")

  snow_fog = "Snow,Fog"
  weather["Weather"] = weather["Weather"].str.replace(snow_fog, "Fog")

  rain_fog = "Rain,Fog"
  weather["Weather"] = weather["Weather"].str.replace(rain_fog, "Fog")


def matchAndSortData(image_data, weather_data, impute=True):

  #clean up weather description and drop unnecessary columns
  CleanWeatherData(weather_data)
  CleanDescription(weather_data)

  if( impute ):
    """
    Impute to account for difference in locations which the image
    data and weather was collected (images of English Bay, weather data from YVR)

    Fills in data from most recent non-blank value
    Good probability that weather from previous hour is the same
    """
    weather_data["Weather"].fillna(method='ffill', inplace=True)

  else:
    weather_data = weather_data[~weather_data["Weather"].isnull()]
  

  weather_times = list(weather_data.timestamp.map(lambda x: x))
  image_data = image_data[image_data.timestamp.isin(weather_times)]

  image_times = list(image_data.timestamp.map(lambda x: x))
  weather_data = weather_data[weather_data.timestamp.isin(image_times)]

  image_data = image_data.sort_values(by='timestamp')
  weather_data = weather_data.sort_values(by='timestamp')

  return image_data, weather_data


def main():

  parser = OptionParser()
  parser.add_option("-p", "--pickle", dest="filename",
                     help="pickle the model to FILE", metavar="FILE")
  parser.add_option("-i", "--impute", dest="impute", action="store_true",
                     help="impute data in csv files")
  parser.add_option("-e","--eq", dest="equalize", action="store_true",
                     help="will use equalized version of the images")
  parser.add_option("-d", "--debug", dest="debug", action="store_true",
                    help="enable debug mode, runs code on subset of data")

  (options, args) = parser.parse_args()

  print("Reading image data...")
  if (options.equalize is not None):
    image_data = ReadImage2GrayScale("./katkam-equalized", rgb=False)
  else:
    image_data = ReadImage2GrayScale("./katkam-scaled", rgb=True)


  if (options.debug is not None):
    weather_data = readMergeData(debug=True)
  else:
    weather_data = readMergeData(debug=False)

  print("Cleaning weather data...")
  print("Spliting data...")

  if (options.impute is not None):
    image_data, weather_data = matchAndSortData(image_data, weather_data)
  else:
    image_data, weather_data = matchAndSortData(image_data, weather_data, impute=False)


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

  if ( options.filename is not None ):
    pickleModel(model, options.filename)



if __name__ == '__main__':
  main()
