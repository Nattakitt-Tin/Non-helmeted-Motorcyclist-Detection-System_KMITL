import cv2
import numpy as np
import sys
import os
re = sys.argv[1]
# vi = []
# re = "sdsdsd\sdsdsdsd\sdsd"

new_re = re.replace('\\','/')
# print("ori : "+re)
print(new_re)

vidcap = cv2.VideoCapture(new_re)
success,image = vidcap.read()
if success:
  success,image = vidcap.read()
  re = cv2.resize(image,(300,200))
  cv2.imwrite("frame.png", re)
  print(image.shape[0])
  print(image.shape[1])
  

# os.mkdir(new_re)
sys.stdout.flush()
