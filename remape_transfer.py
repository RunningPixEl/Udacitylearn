import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread('C:\Users\kangk\Desktop/13.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.resize(gray, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_NEAREST)
mapx = np.zeros(gray.shape, np.float32)
mapy = np.zeros(gray.shape, np.float32)

print gray.shape, mapx.shape,gray.shape[0],gray.shape[1]
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        mapx[i,j] = gray.shape[1] - j
        mapy[i,j] = i
dst_img = cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow("i", dst_img)
cv2.waitKey()
