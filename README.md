## Deployed version

Precomputed models are deployed on heroku: https://eng-bay-weather-model.herokuapp.com/

This is a web application that takes an image of english bay from the user and gives a prediction based on the models below.

The github link for this application is here: https://github.com/selh/ML-Weather

---
## Overview

Scripts to generate the models and print the classification report to terminal. Main python scripts are: 
1) WeatherClassifier.py
2) TimeofDay.py 
3) ReadData.py (Shared functions)
4) image_preprocess.py (needs OpenCV to run, equalizes images)

---
## How to Run:
To run model that classifies images into their corresponding weather conditions:

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

