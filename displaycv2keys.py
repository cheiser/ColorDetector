import cv2
import numpy as np
# Create a black image
img = np.zeros((512,512,3), np.uint8)

cv2.imshow('image',img)
while(True):
    x = cv2.waitKey(-1) & 0xFF
    print x
    if x == 27:
        break
