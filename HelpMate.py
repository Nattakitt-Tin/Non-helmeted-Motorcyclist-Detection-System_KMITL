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
print('load model')
general_model = InceptionResNetV2(weights='imagenet')
print('Custom model loaded complete')
helmet_model = load_model("inceptionResNetV2_BEST.h5")
MOG2 = cv2.createBackgroundSubtractorMOG2(varThreshold=16,history=500,detectShadows=True)
print('Helmet model loaded complete')

helmet_count = 0
no_helmet_count = 0
extra_top = 10

frame = None
x1,y1,x2,y2 = 0,0,0,0
mouse_down = False
crop = False
scale = 0.7

class Person:
    def __init__(self, x, y, w, h, n, life):
        self.xc = int(x+(w/2))
        self.yc = int(y+(h/2))
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.min_y = int(y)
        self.first_y = self.yc
        self.n = n
        self.life = int(life)
        
        self.R = random.randint(0,255)
        self.G = random.randint(0,255)
        self.B = random.randint(0,255)
        self.isSaved = False
        self.direction = 0

    def update(self, frame, x, y, w, h, bottom,the_line):
        if self.yc - self.first_y < 0:
            self.direction = -1 #up
        else:
            self.direction = 1 # down
        self.life = 5
        if self.direction < 0:
            if y > self.min_y:
                y = self.min_y
            else:
                self.min_y = y
        
        self.x = int(x)
        self.y = int(y)
        self.w = w
        self.h = h
        self.xc = int(x+(w/2))
        self.yc = int(y+(h/2))
        
        if the_line < y+h < bottom: #object float off bottom
            self.saveImg(frame, x, y, w, h)
        
    def saveImg(self, frame, x, y, w, h):
        if not self.isSaved and self.direction == -1: # is up
            global helmet_count, no_helmet_count
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

                    img_flip = cv2.flip(img, 1)
                    preds_flip = helmet_model.predict([[img_flip]])
                    helmet_flip = preds_flip[0][1]
                    
                    spt = current_video.split('/')
                    spt = spt[-1]
                    spt = spt.split('.')
                    spt = spt[0]
                    result_path = ''
                    if helmet > 0.5 and helmet_flip > 0.5: #helmet
                        helmet_count += 1
                        result_path = "extracted/helmet/"+spt+"#"+str(helmet_count)
                    else: 
                        no_helmet_count += 1
                        result_path = "extracted/no_helmet/"+spt+"#"+str(no_helmet_count)
                    
                    helmet = str(helmet)
                    helmet = helmet[:4]
                    helmet_flip = str(helmet_flip)
                    helmet_flip = helmet_flip[:4]
                    result_path += " ["+ helmet + '] [' + helmet_flip +"].jpg"

                    cv2.imwrite(result_path,bike_img)
                    found = True
            self.isSaved = True

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

def main(video_name_list, bike_h, max_speed_ratio, show=True, real_fps=False):
    global current_video, progress, crop 

    left,right,top,bottom = 0,0,0,0
    cv2.namedWindow("BGR")
    cv2.setMouseCallback("BGR", mouse_drawing)

    for video_name in video_name_list:
        current_video = video_name
        cap = cv2.VideoCapture(video_name)
        obj_number = 0
        progress = 0
        p_list = []
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
                # cv2.destroyAllWindows()
                break
        fps = None
        delay = None
        the_line = int((bottom-top)*0.66)
        while True:
            _, frame = cap.read()
            if frame is None:
                print('Video Ended, Closing Process')
                break
            if fps is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                print('fps:',fps)
                delay = int(1000/fps) if real_fps else 1
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cmin = '0' + str((current//int(fps))//60); cmin = cmin[-2:]
            csec = '0' + str((current//int(fps))%60); csec = csec[-2:]
            tmin = '0' + str((total//int(fps))//60); tmin = tmin[-2:]
            tsec = '0' + str((total//int(fps))%60); tsec = tsec[-2:]
            time_progress = ' (' + cmin + ':' + csec + '/' + tmin + ':' + tsec + ') '
            sys.stdout.write("Progress: "+ "{0:.3f}".format(100*current/total)+'%' + time_progress)
            sys.stdout.flush()
            sys.stdout.write("\b" * (100))
            # print('the line is', the_line)
            
            frame = frame[top:bottom, left:right]
            frame_copy = frame.copy()
            disp_frame = frame.copy()
            if mouse_down:
                the_line = y2
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
                if h > bike_h and h > w:
                    xc, yc = x+(w/2), y+(h/2)
                    exist = False
                    for p in p_list:
                        accepted_diff =  int(bike_h*max_speed_ratio)
                        if -accepted_diff < xc-p.xc < accepted_diff and -accepted_diff < yc-p.yc < accepted_diff: # center around some object in previous frame, check if its same object
                            p.update(frame_copy, x, y, w, h, frame.shape[0], the_line)
                            exist = True
                            break
                    if not exist:
                        p_list.append(Person(x, y, w, h, obj_number, fps)) # create new person
                        obj_number+=1
                    if show:
                        for p in p_list:
                            cv2.rectangle(disp_frame, (p.x, p.y), (p.x + p.w, p.y + p.h), (p.B, p.G, p.R), 2)
            for p in p_list.copy():
                p.life -= 1
                if p.life == 0:
                    p_list.remove(p)
                p.updated = False
                if show:
                    cv2.putText(disp_frame,str(p.n),(p.xc,p.yc),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            # cv2.drawContours(frame, ctrs, -1, (0, 0, 255), 1)
            if show:
                width = int(disp_frame.shape[1]*scale)
                height = int(disp_frame.shape[0]*scale)
                cv2.line(disp_frame, (0,the_line), (right,the_line), (69,255,36), 2)
                cv2.imshow('BGR', cv2.resize(disp_frame, (width, height)))
                cv2.imshow('Binary', cv2.resize(binary, (width, height)))
                key = cv2.waitKey(delay) ################################# DELAY IS HERE ####################################
                if key == 32:
                    # print("space")
                    key = cv2.waitKey(0)
                if key == 44: # <
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)-100)
                if key == 46: # >
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+100)
                if key == 27:
                    # print("esc")
                    break
            

        cap.release()
        cv2.destroyAllWindows()


main(["C:/Users/59010093/Desktop/Project4D/Non-helmeted-Motorcyclist-Detection-System_KMITL/video/d1.avi"], bike_h=100, max_speed_ratio=2, real_fps=False, show=True)
