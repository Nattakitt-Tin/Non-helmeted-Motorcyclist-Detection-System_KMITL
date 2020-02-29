import cv2
 
f = cv2.VideoCapture('../video/a9.mp4')
 
rval, frame = f.read()
# cv2.imwrite('first_frame.jpg', frame)
while True:
    cv2.imshow("Title of Popup Window", frame)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
