## Deployed version

Precomputed models are deployed on heroku: https://eng-bay-weather-model.herokuapp.com/

This is a web application that takes an image of english bay from the user and gives a prediction based on the models below.

The github link for this application is here: https://github.com/selh/ML-Weather

---
## Overview

WeatherClassifier.py generates a model that classifies weather into categories such as Rain, Clear, Cloudy... The mode was trained using images of English Bay and data from GHCN. TimeofDay.py generates a model to tell the time of day based on the same data. The models were trained using RandomForestClassifier on black and white images. The images were first run through a pre-processing step using image_preprocess.py in order to equalize the contrast in the images and eliminate lighting variations. Note that these models are only trained on weather data collected at YVR-airport and images of English Bay and may not work on other images of the sky.

### Cleaning Process 

The YVR dataset description column had many null values and ambiguous weather descriptions. To clean the description, I used a regex matcher to remove all extraneous words such as Mostly, Mainly, Moderate, Heavy, etc. This cleaning process can be found in WeatherClassifier.py, function name CleanDescription().  After I cleaned the descriptions, I imputed the values in order to fill the null rows with the most recent observed value. I did this in part to account for the slight location differences of the collected image and weather data. In addition, I also decided to remove certain weather conditions such as “Drizzle” and “Thunderstorms” as they had no perceivable differences with those labelled as rain or cloudy in the images provided.

Scripts to generate the models and print the classification report to terminal. Main python scripts are: 
1) WeatherClassifier.py
2) TimeofDay.py 
3) ReadData.py (Shared functions)
4) image_preprocess.py (needs OpenCV to run, equalizes images)

---
## How to Run:
To run model that classifies images into their corresponding weather conditions:
(Original image files not included, please refer to sample_imgs folder to view original format)

Recommended usage:
``` python3 WeatherClassifier.py -d -e ```

How I created my model:
``` python3 WeatherClassifier.py -e -i -pweatherml ```

```
Usage: WeatherClassifier.py [options]

Options:
  -h, --help            show this help message and exit
  -p FILE, --pickle=FILE
                        pickle the model to FILE
  -i, --impute          impute data in csv files
  -e, --eq              will use equalized version of the images
  -d, --debug           enable debug mode, runs code on subset of data
```

To run model that classifies images into their corresponding time period (eg. morning, evening, night)

Recommended usage:
``` python3 TimeofDay.py -d -e ```

How I created my model:
``` python3 TimeofDay.py -e -ptimeofday ```

```
Usage: TimeofDay.py [options]

Options:
  -h, --help            show this help message and exit
  -p FILE, --pickle=FILE
                        pickle the model to FILE
  -e, --eq              will use equalized version of the images
  -d, --debug           enable debug mode, runs code on subset of data
```

To run OpenCV script simply use:
``` python3 image_preprocess.py ```
It will automatically write the equalized images to a folder called katkam-equalized/

---
## Requirements:
Python 3.5.2 

The following python modules are required to run the scripts
1. sklearn 
2. numpy 1.12.1
3. pandas 0.20.3
4. scipy 0.19.1
5. pickle

if running image preprocessing:
6. cv2

---
## Photo Credits:
http://www.katkam.ca/

