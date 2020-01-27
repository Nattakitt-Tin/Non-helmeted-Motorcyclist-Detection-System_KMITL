from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
camera = cv2.VideoCapture(r'F:\project helmet\000.mp4')


detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed="fast")

def get_bike(frame_number, output_array, output_count):
    if frame_number % 5 != 0:
        return
    _,image = camera.read()
    c = 0
    for o in output_array:
        if o['name'] == 'person':
            pos = o['box_points']
            
            y1 = pos[0]
            y2 = pos[2]
            x1 = pos[1]
            x2 = pos[3]
            print('frame',frame_number, x1,',',x2,',',y1,',',y2)

            crop = image[x1:x2, y1:y2]
            cv2.imwrite("crop/f"+str(frame_number)+"_c"+str(c)+".jpg", crop) 
            c+=1


#def getFrame(sec):
#    camera.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#    hasFrames,image = camera.read()
#    if hasFrames:
#        crop_img = image[100:200, 100:x200]
#        cv2.imwrite("image"+str(count)+".jpg", crop_img)     
#    return hasFrames
#sec = 0
#frameRate = 1 
#count=1
#success = getFrame(sec)
#while success:
#    count = count + 1
#    sec = sec + frameRate
#    sec = round(sec, 2)
#    success = getFrame(sec)

video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "camera_detected_1")
                                , frames_per_second=29, log_progress=True, per_frame_function=get_bike)
print(video_path)
