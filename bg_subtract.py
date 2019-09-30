import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

class Person:
    def __init__(self, x, y, n):
        self.x = int(x)
        self.y = int(y)
        self.n = n
        self.life = 5
        self.isSaved = False
        self.direction = 0
    def update(self, x, y):
        print(y, 'and', self.y)
        if y - self.y < 0:
            self.direction = -1 #up
        else:
            self.direction = 1 # down
        self.life = 5
        self.x = int(x)
        self.y = int(y)
    def saveImg(self, frame, x, y, w, h):
        print('contest for save', self.direction)
        if True: # self.direction == -1: is up
            img = cv2.resize(frame_r[y:y+h, x:x+w], (299,299)) 
            imgx = image.img_to_array(img)
            imgx = np.expand_dims(imgx, axis=0)
            imgx = preprocess_input(imgx)

            preds = model.predict(imgx)
            found = False
            t3 = decode_predictions(preds, top=3)[0]
            print(t3)
            for result in t3:
                if result[1] == 'motor_scooter' and result[2] > 0.1:
                    cv2.imwrite("crop_image2/bike_"+str(self.n)+".jpg",frame[y:y+h, x:x+w])
                    found = True
            if found:
                print(self.n,'-',t3)
        self.isSaved = True
model = InceptionResNetV2(weights='imagenet')
cap = cv2.VideoCapture('cctv3.mp4')
KNN = cv2.createBackgroundSubtractorMOG2(varThreshold=100,history=300)
number = 0
p_list = []
while True:
    # print("- - - - - - - - - -")
    _, frame_r = cap.read()
    frame = frame_r.copy()

    tresh = KNN.apply(frame)
    _, tresh = cv2.threshold(tresh,100,255,cv2.THRESH_BINARY)
    # kernel = np.ones((3,3),np.uint8)
    # tresh = cv2.morphologyEx(tresh, cv2.MORPH_CLOSE, kernel)
    # tresh = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel)
    # kernel = np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8)
    # tresh = cv2.dilate(tresh,kernel,iterations=5)

    ctrs, hier = cv2.findContours(tresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    for i, ctr in enumerate(ctrs):
        x, y, w, h = cv2.boundingRect(ctr)  # get xywh from points
        if h > 200 and y > frame.shape[0]*0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            x0, y0 = x+(w/2), y+(h/2)
            exist = False
            for p in p_list:
                # print(x0, p.x," <> ",y0, p.y)
                # print('x',x0-p.x,'<-> y',y0-p.y)
                diff =  int(frame.shape[0]*0.20)
                if -diff < x0-p.x < diff and -diff < y0-p.y < diff: # center around some object in previous frame
                    print("update", end = " ")
                    p.update(x0,y0)
                    exist = True
                    if not p.isSaved:
                        p.saveImg(frame_r, x, y, w, h)
                    break
            if not exist:
                print("new",number)
                p_list.append(Person(x0, y0, number)) # create new person
                number+=1


    for p in p_list.copy():
        p.life -= 1
        if p.life == 0:
            # print("remove",p.n)
            p_list.remove(p)
        p.updated = False
        cv2.putText(frame,str(p.n),(p.x,p.y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    # cv2.drawContours(frame, ctrs, -1, (0, 0, 255), 1)
    cv2.line(frame, (0, int(frame.shape[0]*0.5)), (frame.shape[1], int(frame.shape[0]*0.5)), (125,125,0), 2)
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