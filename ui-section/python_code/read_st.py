import cv2
import numpy as np
import os
import sys


# # path = "D:/CopyFileCCTV/Chanel66/d/d1.avi"
# out = "F:/project/Non-helmeted-Motorcyclist-Detection-System_KMITL/ui-section/python_code"

path = sys.argv[1]

vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()
if success:
  success,image = vidcap.read()
  re = cv2.resize(image,(300,200))
  cv2.imwrite("frame.png", re)
  print(image.shape[0])
  print(image.shape[1])
  
  # cv2.imshow('image',re)
  # print('Read a new frame: ', success)
  # os.remove("frame.png")
  # cv2.waitKey(0)

# print(path)

