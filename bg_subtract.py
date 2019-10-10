import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

#parameter set
video_name_list = ['a12','a13']
#yes, video name
show = False
#show frame
cpt_range = 50
#distance from bottom we accept to capture (to get the biggest obj)
disc_ratio = 0.3
#Interes area, from 0. to 1. 
trsh = 16
#Background Subtraction Threshold
shadow = True
#Enable shadow detect
hist = 500 
#Background Subtraction Average frame history
extra_top = 10
#additional height to take, prevent a headless biker, in pixel1
bike_h = 200
#expected bike height/width, in pixel
max_speed_ratio = 0.2
# ratio on frame height that we accept when 2 object between frame move
accepted_bike_threshold = 0.1
#accuracy accept
img_num = 2300
#started img number
### -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - ###
current_video = 'xxx'
model = InceptionResNetV2(weights='imagenet')
MOG2 = cv2.createBackgroundSubtractorMOG2(varThreshold=trsh,history=hist,detectShadows=shadow)

class Person:
    def __init__(self, x, y, w, h, n):
        self.x = int(x)
        self.y = int(y)
        self.w = w
        self.h = h
        self.first_y = int(y)
        self.n = n
        self.life = 5
        
        self.isSaved = False
        self.direction = 0

    def update(self, xc, yc, frame, x, y, w, h, bottom):
        if yc - self.first_y < 0:
            self.direction = -1 #up
        else:
            self.direction = 1 # down
        self.life = 5
        self.x = int(xc)
        self.y = int(yc)
        self.w = w
        self.h = h
        if bottom - cpt_range < y+h < bottom: #object float off bottom (default should be 0 without black bar)
            self.saveImg(frame, x, y, w, h)
        
    def saveImg(self, frame, x, y, w, h):
        if not self.isSaved and self.direction == -1: # is up
            global img_num
            global current_video
            y0 = y-extra_top if y-extra_top > 0 else 0
            img = cv2.resize(frame[y0:y+h, x:x+w], (299,299))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgx = image.img_to_array(img)
            imgx = np.expand_dims(imgx, axis=0)
            imgx = preprocess_input(imgx)

            preds = model.predict(imgx)
            found = False
            t3 = decode_predictions(preds, top=3)[0]
            # print('It is', t3)
            
            for result in t3:
                if result[1] == 'motor_scooter' and result[2] > accepted_bike_threshold:
                    cv2.imwrite("extracted/bike_"+str(img_num)+"_"+str(current_video)+".jpg",frame[y0:y+h, x:x+w])
                    found = True
            if found:
                print()
                print('Bike #'+str(img_num),':',t3)
                img_num += 1
            self.isSaved = True

for video_name in video_name_list:
    current_video = video_name
    cap = cv2.VideoCapture('video/'+video_name+'.mp4')
    obj_number = 0
    progress = 0
    p_list = []
    print("Process Started on Video:",video_name+'.mp4')
    _, first = cap.read()
    if first is None:
            print('Nothing to read, Closing Process')
            continue
    while np.average(first) < 10:
        _, first = cap.read()
    first = first[int(first.shape[0]*disc_ratio):,:]
    bottom = first.shape[0]
    while True:
        bottom -= 1
        avg = np.average(first[bottom])
        if avg > 10: 
            break
    while True:
        _, frame = cap.read()
        if frame is None:
            print('Video Ended, Closing Process')
            break
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        sys.stdout.write("Progress: "+ "{0:.3f}".format(100*current/total)+'% ('+str((current//30)//60)+':'+str((current//30)%60)+'/'+str((total//30)//60)+':'+str((total//30)%60)+')')
        sys.stdout.flush()
        sys.stdout.write("\b" * (100))

        
        half_frame = frame.copy()
        half_frame = half_frame[int(half_frame.shape[0]*disc_ratio):,:]
        disp_frame = half_frame.copy()
        binary = MOG2.apply(half_frame)
        _, binary = cv2.threshold(binary,200,255,cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        kernel = np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]],np.uint8)
        binary = cv2.dilate(binary,kernel)

        ctrs, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        for i, ctr in enumerate(ctrs):
            x, y, w, h = cv2.boundingRect(ctr)  # get xywh from points
            if h > bike_h:
                xc, yc = x+(w/2), y+(h/2)
                exist = False
                for p in p_list:
                    # print(x0, p.x," <> ",y0, p.y)
                    # print('x',x0-p.x,'<-> y',y0-p.y)
                    accepted_diff =  int(half_frame.shape[0]*max_speed_ratio)
                    if -accepted_diff < xc-p.x < accepted_diff and -accepted_diff < yc-p.y < accepted_diff: # center around some object in previous frame, check if its same object
                        p.update(xc, yc, half_frame, x, y, w, h, bottom)
                        exist = True
                        break
                if not exist:
                    p_list.append(Person(xc, yc, w, h, obj_number)) # create new person
                    obj_number+=1
                if show:
                    cv2.rectangle(disp_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for p in p_list.copy():
            p.life -= 1
            if p.life == 0:
                # print("remove",p.n)
                p_list.remove(p)
            p.updated = False
            if show:
                cv2.putText(disp_frame,str(p.n),(p.x,p.y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        # cv2.drawContours(frame, ctrs, -1, (0, 0, 255), 1)
        # cv2.line(half_frame, (0, int(half_frame.shape[0]*0.5)), (half_frame.shape[1], int(half_frame.shape[0]*0.5)), (200,0,200), 2)
        if show:
            cv2.imshow('BGR', cv2.resize(disp_frame, (int(half_frame.shape[1]*0.7),int(half_frame.shape[0]*0.7))))
            cv2.imshow('Binary', cv2.resize(binary, (int(half_frame.shape[1]*0.7),int(half_frame.shape[0]*0.7))))
            key = cv2.waitKey(1) ################################# DELAY IS HERE ####################################
            if key == 32:
                # print("space")
                key = cv2.waitKey(0)
            if key == 27:
                # print("esc")
                break

    cap.release()
    cv2.destroyAllWindows()