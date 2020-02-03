import cv2
import numpy as np 
import sys

frame = None
x1,y1,x2,y2 = 0,0,0,0
mouse_down = False
crop = False
scale = 0.7
p = sys.argv[1]
re_p = p.replace('\\','/')

def mouse_drawing(event, x, y, flags, params):
    global x1,y1,x2,y2,mouse_down,crop
    if not crop:
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
            # print(x1,y1,x2,y2)
            print(x1)
            print(y1)
            print(x2)
            print(y2)
            print(re_p)
            sys.stdout.flush()
        if mouse_down:
            x2 = int(x/scale)
            y2 = int(y/scale)
    else:
        if (event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_LBUTTONUP) and not mouse_down:
            crop = False
            
            x1,y1,x2,y2 = 0,0,0,0

cap = cv2.VideoCapture(re_p)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)

while True:
    _, frame = cap.read()
    if (x1 > 0 or x2 > 0 or  y1 > 0 or x1 > 0) and not mouse_down:
        frame = frame[y1:y2, x1:x2]
        crop = True
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    if mouse_down:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0, 0, 255), 2)
    cv2.imshow("Frame", cv2.resize(frame, (width, height)))
    key = cv2.waitKey(10)
    if key == 27:
        break
