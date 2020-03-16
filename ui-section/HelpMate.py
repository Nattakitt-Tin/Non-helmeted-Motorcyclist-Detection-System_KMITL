import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import os

os.system("conda activate gputest")
sys.path.append("C:/Users/59010401/conda/envs/gputest/python.exe")

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

current_video = 'xxx'
print('load model')
general_model = InceptionResNetV2(weights='imagenet')
print('Custom model loaded complete')
helmet_model = load_model("ResNet.h5")
MOG2 = cv2.createBackgroundSubtractorMOG2(varThreshold=16,history=500,detectShadows=True)
print('Helmet model loaded complete')

n_list = sys.argv[1]
n_list = n_list.split(',')
out_f = sys.argv[2]
out_f = out_f.replace('\r','')
# out_f = "C:/Users/59010401/Desktop/extracted"
# print(out_f)
pos1 = int(sys.argv[3])
pos2 = int(sys.argv[4])
pos3 = int(sys.argv[5])
pos4 = int(sys.argv[6])
print(type(pos1))
helmet_count = 0
no_helmet_count = 0
extra_top = 10
ref_area = 0
the_line = 0

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
        self.first_x = self.xc
        self.n = n
        self.life = int(life)
        
        self.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        self.footprint = [(self.xc,self.yc)]
        self.detectToken = 4
        self.status = (255,255,255)
        self.upward = None
        self.rightward = None

    def update(self, frame, x, y, w, h, fps):
        self.life = int(fps)
        self.x = int(x)
        self.y = int(y)
        self.w = w
        self.h = h
        self.xc = int(x+(w/2))
        self.yc = int(y+(h/2))
        self.footprint.append((self.xc,self.yc))
        if self.yc - self.first_y < 0:
            self.upward = True #up
        else:
            self.upward = False # down

        if self.xc - self.first_x < 0:
            self.rightward = False # <<
        else:
            self.rightward = True # >>
        # dist = ((self.yc - self.first_y)**2 + (self.xc - self.first_x)**2)**(0.5)
        if self.detectToken > 0 and self.upward and self.yc > the_line:
            self.detectToken -= 1
            self.status = (0,0,0)
            self.saveImg(frame, x, y, w, h)
        
    def saveImg(self, frame, x, y, w, h):
        global helmet_count, no_helmet_count, current_video, ref_area, the_line
        area = w*h
        # ref_area = area if ref_area == 0 else ref_area
        # if area < ref_area*0.5:
        #     return
        y0 = y-extra_top if y-extra_top > 0 else 0
        bike_img = frame[y0:y+h, x:x+w]
        sqr_img = cv2.resize(bike_img, (299,299))
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sqr_img = image.img_to_array(sqr_img)
        imgx = np.expand_dims(sqr_img, axis=0)
        preprocess_input(imgx) #may use /255. if something go wrong
        preds = general_model.predict(imgx)

        top = decode_predictions(preds, top=5)[0]

        for result in top:
            if result[1] == 'motor_scooter': #and result[2] > 0.1:
                self.detectToken = 0
                the_line = int(the_line*0.5 + self.y*0.5)
                img = cv2.resize(bike_img, (224,224))
                img = img/255.
                if not self.rightward:
                    img = cv2.flip(img, 1)

                preds = helmet_model.predict([[img]])
                helmet = preds[0][1]
                
                spt = current_video.split('/')
                spt = spt[-1]
                spt = spt.split('.')
                spt = spt[0]
                result_path = ''
                if helmet > 0.5: #helmet
                    helmet_count += 1
                    result_path = out_f+"/helmet/"+spt+"#"+str(helmet_count)
                    self.status = (0,255,0)
                else: 
                    helmet = 1 - helmet
                    no_helmet_count += 1
                    result_path = out_f+"/no_helmet/"+spt+"#"+str(no_helmet_count)
                    self.status = (0,0,255)
                
                helmet = str(helmet*100)
                helmet = helmet.split('.')
                a = helmet[0]
                b = helmet[1][:2]
                result_path += " ["+ a + "." + b + "%].jpg"
                cv2.imwrite(result_path,bike_img)
                break
        

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

def main(video_name_list, bike_h, show=True, real_fps=False):
    global current_video, progress, crop ,pos1,pos2,pos3,pos4

    left,right,top,bottom = pos1,pos2,pos3,pos4
    cv2.namedWindow("BGR")
    # cv2.setMouseCallback("BGR", mouse_drawing)

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
        # while not crop:
        #     disp_frame = first.copy()
        #     if mouse_down:
        #         cv2.rectangle(disp_frame,(x1,y1),(x2,y2),(0, 0, 252), 2)
        #     width = int(disp_frame.shape[1]*scale)
        #     height = int(disp_frame.shape[0]*scale)
        #     cv2.imshow('BGR', cv2.resize(disp_frame, (width, height)))
        #     key = cv2.waitKey(1)
        #     if (x1 > 0 or x2 > 0 or  y1 > 0 or y2 > 0) and not mouse_down:
        #         left,right,top,bottom = x1,x2,y1,y2
        #         crop = True
        #         # cv2.destroyAllWindows()
        #         break
        fps = None
        delay = None
        # the_line = int((bottom-top)*0.7)
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
            # if mouse_down:
            #     the_line = y2
            binary = MOG2.apply(frame_copy)
            _, binary = cv2.threshold(binary,200,255,cv2.THRESH_BINARY)
            kernel = np.ones((5,5),np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            kernel = np.array([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]],np.uint8)
            binary = cv2.dilate(binary,kernel)

            ctrs, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            detected_p = []
            for ctr in ctrs:
                x, y, w, h = cv2.boundingRect(ctr)  # get xywh from points
                in_frame = x > 0 and x+w < frame.shape[1] and y > 0 and y+h < frame.shape[0]
                if h > bike_h and h > w and in_frame:
                    xc, yc = x+(w/2), y+(h/2)
                    exist = False
                    min_diff = 9999
                    nearest_p = None
                    for p in p_list:
                        if p not in detected_p:
                            diff = ((xc-p.xc)**2 + (yc-p.yc)**2)**(0.5)
                            if diff < min_diff and diff < bike_h*2:
                                min_diff = diff
                                nearest_p = p
                                exist = True
                    if exist:
                        nearest_p.update(frame_copy, x, y, w, h,fps)
                        detected_p.append(nearest_p)
                    else:
                        new_person = Person(x, y, w, h, obj_number, fps)
                        p_list.append(new_person) # create new person
                        detected_p.append(new_person)
                        obj_number += 1
            for p in p_list.copy():
                p.life -= 1
                if p.life == 0:
                    p_list.remove(p)
            if show:
                cv2.line(disp_frame, (0,the_line), (right,the_line), (0,100,255), 2)
                for p in p_list:
                    cv2.rectangle(disp_frame, (p.x, p.y), (p.x + p.w, p.y + p.h), p.color, 3)
                    cv2.putText(disp_frame,str(p.n),(p.xc,p.yc),cv2.FONT_HERSHEY_SIMPLEX,1,p.status,2)
                    last,curr = None,None
                    for point in p.footprint:
                        curr = point
                        if curr is not None and last is not None:
                            cv2.line(disp_frame, curr, last, p.color,3)
                        last = curr
                width = int(disp_frame.shape[1]*scale)
                height = int(disp_frame.shape[0]*scale)
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

file_names = n_list
main(file_names, bike_h=100, real_fps=False, show=True)
