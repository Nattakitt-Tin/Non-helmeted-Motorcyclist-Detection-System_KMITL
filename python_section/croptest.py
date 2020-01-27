from imageai.Detection import ObjectDetection
import os
import cv2

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed="fast")


cap = cv2.VideoCapture(r'F:\project helmet\111.mp4')
KNN = cv2.createBackgroundSubtractorMOG2(varThreshold=200,history=60)
fr = 0
while True:
    _, frame_r = cap.read()
    fr = fr + 1
    if fr%1 == 0:
        ylow = 0
        frame = frame_r.copy()
        mask = KNN.apply(frame)
        _, tresh = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
        ctrs, hier = cv2.findContours(tresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        for i, ctr in enumerate(ctrs):
            x, y, w, h = cv2.boundingRect(ctr)  # get xywh from points
            if w > 30 and h > 30:
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                if y > ylow:
                    ylow = y
        if ylow > frame.shape[0]*0.85:
            detections = detector.detectObjectsFromImage(input_type= 'array' ,input_image=frame_r, output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)
            n = 0
            for eachObject in detections:
                co = eachObject["box_points"]
                x1,x2,y1,y2 = co[0],co[2],co[1],co[3]
                if eachObject["name"] == "person":
                    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
                    print("--------------------------------")
                    # cv2.imshow("Person number "+str(n), cap[y1:y2, x1:x2]) # cropped image
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    n+=1
                    print("{",x2-x1, y2-y1,"}")
    cv2.imshow('Frame', frame)
    cv2.imshow('KNN',tresh)
    key = cv2.waitKey(15)
    if key == 32:
        print("space")
        key = cv2.waitKey(0)
    if key == 27:
        print("esc")
        break
cap.release()
cv2.destroyAllWindows()


while(video.isOpened()):
    z, cap = video.read()
    cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    
    cv2.imshow("video", cap) #original video
    
    key = cv2.waitKey(1)
    if key == 32:
        for _ in range(30):
            video.read()
    if key == 27:
        break
    

    
