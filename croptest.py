from imageai.Detection import ObjectDetection
import os
import cv2

execution_path = os.getcwd()
video = cv2.VideoCapture(r'F:\project helmet\000.mp4')
z, cap = video.read()
# cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed="fast")

while(video.isOpened()):
    detections = detector.detectObjectsFromImage(input_type= 'array' ,input_image=cap, output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)
    n = 0
    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
        co = eachObject["box_points"]
        cv2.imshow(str(n), cap[co[1]:co[3], co[0]:co[2]])
        n+=1 
    z, cap = video.read()
    if cv2.waitKey(1) == 32:
        break
