import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

class Person:
    def __init__(self, x, y, w, h, n):
        self.x = int(x)
        self.y = int(y)
        self.w = w
        self.h = h
        self.size = w*h
        self.first_y = int(y)
        self.n = n
        self.life = 5
        
        self.isSaved = False
        self.direction = 0

    def update(self, xc, yc, frame_r, x, y, w, h):
        # print('update #', self.n)
        if yc - self.first_y < 0:
            self.direction = -1 #up
        else:
            self.direction = 1 # down
        self.life = 5
        self.x = int(xc)
        self.y = int(yc)
        self.w = w
        self.h = h
        if y+h < 840: #object float off bottom (default should be 0 without black bar)
            self.saveImg(frame_r, x, y, w, h)
        self.size = w*h # new size
        

    def saveImg(self, frame, x, y, w, h):
        if not self.isSaved and self.direction == -1: # is up
            img = cv2.resize(frame_r[y:y+h, x:x+w], (299,299))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgx = image.img_to_array(img)
            imgx = np.expand_dims(imgx, axis=0)
            imgx = preprocess_input(imgx)

            preds = model.predict(imgx)
            found = False
            t3 = decode_predictions(preds, top=3)[0]
            print('It is', t3)
            for result in t3:
                if result[1] == 'motor_scooter' and result[2] > 0.1:
                    cv2.imwrite("crop_image2/bike_"+str(self.n)+".jpg",frame[y:y+h, x:x+w])
                    found = True
            if found:
                print(self.n,'-',t3)
            self.isSaved = True

model = InceptionResNetV2(weights='imagenet')
cap = cv2.VideoCapture('cctv3.mp4')
KNN = cv2.createBackgroundSubtractorMOG2(varThreshold=10,history=600)
number = 0
p_list = []
while True:
    # print("- - - - - - - - - -")
    _, frame_r = cap.read()
    frame = frame_r.copy()

    tresh = KNN.apply(frame)
    _, tresh = cv2.threshold(tresh,200,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    tresh = cv2.morphologyEx(tresh, cv2.MORPH_CLOSE, kernel)
    tresh = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel)
    # tresh = cv2.dilate(tresh,kernel)
    # kernel = np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8)
    # tresh = cv2.dilate(tresh,kernel,iterations=5)

    ctrs, hier = cv2.findContours(tresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)  # get xywh from points
        if h > 200 and y > frame.shape[0]*0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            xc, yc = x+(w/2), y+(h/2)
            exist = False
            for p in p_list:
                # print(x0, p.x," <> ",y0, p.y)
                # print('x',x0-p.x,'<-> y',y0-p.y)
                diff =  int(frame.shape[0]*0.20)
                if -diff < xc-p.x < diff and -diff < yc-p.y < diff: # center around some object in previous frame, check if its same object
                    p.update(xc, yc, frame_r, x, y, w, h)
                    exist = True
                    break
            if not exist:
                print("new",number)
                p_list.append(Person(xc, yc, w, h, number)) # create new person
                number+=1


    for p in p_list.copy():
        p.life -= 1
        if p.life == 0:
            # print("remove",p.n)
            p_list.remove(p)
        p.updated = False
        cv2.putText(frame,str(p.n),(p.x,p.y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    # cv2.drawContours(frame, ctrs, -1, (0, 0, 255), 1)
    cv2.line(frame, (0, int(frame.shape[0]*0.5)), (frame.shape[1], int(frame.shape[0]*0.5)), (200,0,200), 2)
    cv2.imshow('Frame', cv2.resize(frame, (int(frame.shape[1]*0.7),int(frame.shape[0]*0.7))))
    cv2.imshow('KNN', cv2.resize(tresh, (int(frame.shape[1]*0.7),int(frame.shape[0]*0.7))))
    key = cv2.waitKey(1)
    if key == 32:
        # print("space")
        key = cv2.waitKey(0)
    if key == 27:
        # print("esc")
        break
cap.release()
cv2.destroyAllWindows()