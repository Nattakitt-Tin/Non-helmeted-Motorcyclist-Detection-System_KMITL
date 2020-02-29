import cv2 as cv2
import base64
import sys
import os

source = cv2.imread("knn.png", cv2.IMREAD_GRAYSCALE)
success, encoded_image = cv2.imencode('.png', source)
content = encoded_image.tobytes()
print(base64.b64encode(content).decode('ascii'))
sys.stdout.flush()
