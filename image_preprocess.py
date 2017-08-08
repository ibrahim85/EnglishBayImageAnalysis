#script to do histogram equalization on images.

import numpy as np
import glob
import cv2
import re


def main():

  for path in sorted(glob.glob("./katkam-scaled/*")):

    filename = re.search("[^/]+$", path).group()
    img = cv2.imread(path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(img)

    cv2.imwrite('./katkam-equalized/' + filename + '.jpg', equalized)
  



if __name__ == '__main__':
  main()