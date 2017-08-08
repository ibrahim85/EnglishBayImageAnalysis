'''
  Trains and predicts the time of day based on the image of english bay
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ReadData import ReadImage2GrayScale, dataFrame2Array, readMergeData, CleanWeatherData, pickleModel
from optparse import OptionParser
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 

def matchAndSortData(image_data, weather_data):

  #clean up weather description and drop unnecessary columns
  CleanWeatherData(weather_data)

  weather_times = list(weather_data.timestamp.map(lambda x: x))
  image_data = image_data[image_data.timestamp.isin(weather_times)]

  image_times = list(image_data.timestamp.map(lambda x: x))
  weather_data = weather_data[weather_data.timestamp.isin(image_times)]

  image_data = image_data.sort_values(by='timestamp')
  weather_data = weather_data.sort_values(by='timestamp')

  return image_data, weather_data


def makeHourColumn(weather_data):
  weather_data["hour"] = pd.DatetimeIndex(weather_data["timestamp"]).hour


def makeTimeRange(timestamp):
  hour = timestamp.hour
  if hour <= 12:
    return "Morning"

  if hour <= 16:
    return "Afternoon"

  if hour <= 19:
    return "Evening"

  return "Night"


def main():

  parser = OptionParser()
  parser.add_option("-p", "--pickle", dest="filename",
                     help="pickle the model to FILE", metavar="FILE")
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

  image_data, weather_data = matchAndSortData(image_data, weather_data)
  X_train_img, X_test_img, y_train, y_test = train_test_split(image_data, weather_data, train_size=0.9)

  X_train = np.zeros((X_train_img.shape[0],49152), dtype=np.float32)
  X_test  = np.zeros((X_test_img.shape[0],49152), dtype=np.float32)

  dataFrame2Array(X_train, X_train_img)
  dataFrame2Array(X_test, X_test_img)

  print("Matching Time of Day...")

  y_train.loc[:,'timeofday'] = y_train.timestamp.apply(makeTimeRange)
  y_test.loc[:,'timeofday'] = y_test.timestamp.apply(makeTimeRange)

  model = RandomForestClassifier(n_estimators=50)

  model.fit(X_train, y_train.timeofday)
  #print(model.score(X_test, y_test.timeofday))
  y_predicted = model.predict(X_test)
  print(classification_report(y_test.timeofday,y_predicted))

  if ( options.filename is not None ):
    pickleModel(model, options.filename)

if __name__ == '__main__':
  main()