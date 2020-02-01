import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions


current_video = 'xxx'
print('load model')
general_model = InceptionResNetV2(weights='imagenet')
print('InceptionResnetV2 loaded complete')
helmet_model = load_model("incV3_final.h5")
MOG2 = cv2.createBackgroundSubtractorMOG2(varThreshold=16,history=500,detectShadows=True)
print('helmet model loaded complete')

img_num = 0
extra_top = 10

frame = None
x1,y1,x2,y2 = 0,0,0,0
mouse_down = False
crop = False
scale = 0.7

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
        if bottom - 50 < y+h < bottom: #object float off bottom (default should be 0 without black bar)
            self.saveImg(frame, x, y, w, h)
        
    def saveImg(self, frame, x, y, w, h):
        if not self.isSaved and self.direction == -1: # is up
            global img_num
            global current_video
            y0 = y-extra_top if y-extra_top > 0 else 0
            bike_img = frame[y0:y+h, x:x+w]
            sqr_img = cv2.resize(bike_img, (299,299))
            
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # cv2.imshow('img',sqr_img)
            sqr_img = image.img_to_array(sqr_img)
            imgx = np.expand_dims(sqr_img, axis=0)
            preprocess_input(imgx) #may use /255. if something go wrong
            preds = general_model.predict(imgx)

            found = False
            t3 = decode_predictions(preds, top=3)[0]

            for result in t3:
                if result[1] == 'motor_scooter' and result[2] > 0.1:
                    # sqr_img = sqr_img/255.
                    img = cv2.resize(bike_img, (299,299))
                    img = img/255.
                    preds = helmet_model.predict([[img]])
                    helmet = preds[0][1]
                    result_path = "extracted/helmet/bike_"+str(img_num)+"_"+str(current_video)+".jpg"
                    if helmet < 0.5:
                        # print('+ + + no helmet! + + +',helmet)
                        result_path = "extracted/no_helmet/bike_"+str(img_num)+"_"+str(current_video)+".jpg"
                    cv2.imwrite(result_path,bike_img)
                    found = True
            if found:
                # print()
                # print('Bike #'+str(img_num),':',t3)
                img_num += 1
            self.isSaved = True

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
        if mouse_down:
            x2 = int(x/scale)
            y2 = int(y/scale)
    else:
        if event == cv2.EVENT_RBUTTONUP and not mouse_down:
            crop = False
            x1,y1,x2,y2 = 0,0,0,0

def main(video_name_list, bike_h, max_speed_ratio, show=True):
    global current_video, progress, crop
    cv2.namedWindow("BGR")
    cv2.setMouseCallback("BGR", mouse_drawing)
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
        while np.average(first) < 10: #skip black frame
            _, first = cap.read()
        
        while True:
            if not mouse_down:
                _, frame = cap.read()
            if frame is None:
                print('Video Ended, Closing Process')
                break
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            sys.stdout.write("Progress: "+ "{0:.3f}".format(100*current/total)+'% ('+str((current//30)//60)+':'+str((current//30)%60)+'/'+str((total//30)//60)+':'+str((total//30)%60)+')')
            sys.stdout.flush()
            sys.stdout.write("\b" * (100))

            if (x1 > 0 or x2 > 0 or  y1 > 0 or x2 > 0) and not mouse_down:
                frame = frame[y1:y2, x1:x2]
                crop = True
            frame_copy = frame.copy()
            disp_frame = frame.copy()
            if mouse_down:
                cv2.rectangle(disp_frame,(x1,y1),(x2,y2),(0, 0, 255), 2)

            binary = MOG2.apply(frame_copy)
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
                        accepted_diff =  int(frame_copy.shape[0]*max_speed_ratio)
                        if -accepted_diff < xc-p.x < accepted_diff and -accepted_diff < yc-p.y < accepted_diff: # center around some object in previous frame, check if its same object
                            p.update(xc, yc, frame_copy, x, y, w, h, frame.shape[0])
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
                    p_list.remove(p)
                p.updated = False
                if show:
                    cv2.putText(disp_frame,str(p.n),(p.x,p.y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            # cv2.drawContours(frame, ctrs, -1, (0, 0, 255), 1)
            # cv2.line(half_frame, (0, int(half_frame.shape[0]*0.5)), (half_frame.shape[1], int(half_frame.shape[0]*0.5)), (200,0,200), 2)
            if show:
                width = int(disp_frame.shape[1]*scale)
                height = int(disp_frame.shape[0]*scale)
                cv2.imshow('BGR', cv2.resize(disp_frame, (width, height)))
                # cv2.imshow('Binary', cv2.resize(binary, (width, height)))
                key = cv2.waitKey(1) ################################# DELAY IS HERE ####################################
                if key == 32:
                    # print("space")
                    key = cv2.waitKey(0)
                if key == 27:
                    # print("esc")
                    break
        cap.release()
        cv2.destroyAllWindows()


main(video_name_list=['b2'], bike_h=200, max_speed_ratio=0.2)