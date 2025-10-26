import cv2
import numpy as np

# Translation matrix: move right by 50px, down by 100px
M = np.float32([
    [1, 0, 50],
    [0, 1, 100]
])

img = cv2.imread('Dataset_CvDl_Hw1/Q1_Image/1.bmp')
shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

cv2.imshow('Shifted', shifted)
cv2.waitKey(0)
