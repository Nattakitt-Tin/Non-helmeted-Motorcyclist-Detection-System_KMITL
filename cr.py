import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

current_video = 'xxx'

frame = None
x1,y1,x2,y2 = 0,0,0,0
mouse_down = False
crop = False
scale = 0.7

def mouse_drawing(event, x, y, flags, params):
    global x1,y1,x2,y2,mouse_down,crop
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = int(x/scale)
        y1 = int(y/scale)
        mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        if x1 > x2:
            x1,x2 = x2,x1
        if y1 > y2:
            y1,y2 = y2,y1
    if mouse_down:
        x2 = int(x/scale)
        y2 = int(y/scale)

def main(video_name_list,max_speed_ratio, show=True, real_fps=False):
    global current_video, crop 

    left,right,top,bottom = 0,0,0,0
    cv2.namedWindow("BGR")
    cv2.setMouseCallback("BGR", mouse_drawing)

    for video_name in video_name_list:
        current_video = video_name
        cap = cv2.VideoCapture(video_name)
        print("Process Started on Video:",video_name)
        _, first = cap.read()
        if first is None:
                print('Nothing to read, Closing Process')
                continue
        while np.average(first) < 10: #skip black frame
            _, first = cap.read()
        while not crop:
            disp_frame = first.copy()
            if mouse_down:
                cv2.rectangle(disp_frame,(x1,y1),(x2,y2),(0, 0, 252), 2)
            width = int(disp_frame.shape[1]*scale)
            height = int(disp_frame.shape[0]*scale)
            cv2.imshow('BGR', cv2.resize(disp_frame, (width, height)))
            key = cv2.waitKey(1)
            if (x1 > 0 or x2 > 0 or  y1 > 0 or y2 > 0) and not mouse_down:
                left,right,top,bottom = x1,x2,y1,y2
                crop = True
                print(left,right,top,bottom)

main(["F:/project/Non-helmeted-Motorcyclist-Detection-System_KMITL/video/d1.avi"], max_speed_ratio=2, real_fps=False, show=True)
