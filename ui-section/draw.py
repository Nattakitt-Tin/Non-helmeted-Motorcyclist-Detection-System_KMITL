import cv2
img = cv2.imread("knn.png")

ix = -1
iy = -1
drawing = False

def draw_reactangle_with_drag(event, x, y, flags, param):
    global ix, iy, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
            
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img2 = cv2.imread("knn.png")
            cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(0,255,0),thickness=2)
            img = img2
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img2 = cv2.imread("knn.png")
        cv2.rectangle(img2, pt1=(ix,iy), pt2=(x, y),color=(0,255,0),thickness=2)
        img = img2
        print("x = "+str(ix)+" - "+str(x))
        print("y = "+str(iy)+" - "+str(y))
        
cv2.namedWindow(winname= "Title of Popup Window")
cv2.setMouseCallback("Title of Popup Window", draw_reactangle_with_drag)

while True:
    cv2.imshow("Title of Popup Window", img)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()