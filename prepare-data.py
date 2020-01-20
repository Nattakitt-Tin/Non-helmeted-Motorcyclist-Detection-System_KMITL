import glob
import cv2 as cv
img_num = 1
dim = (224, 224)
path = glob.glob("dataset/test/no/*.jpg")
for img in path:
    n = cv.imread(img,cv.IMREAD_UNCHANGED)
    # print(str(img_num))
    print(img)
    resized = cv.resize(n, dim, interpolation = cv.INTER_AREA)
    ok = cv.imwrite('224data/test/no/no_'+str(img_num)+'.jpg',resized)
    img_num += 1







