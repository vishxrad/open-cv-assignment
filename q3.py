import cv2
import numpy as np

# Load images in grayscale
imgL = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imgL.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imgR.png', cv2.IMREAD_GRAYSCALE)

# Check if images were loaded correctly
if imgL is None or imgR is None:
    print("Error: Could not read one or both images.")
else:
    stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)

    disparity = stereo.compute(imgL, imgR)

    
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Disparity Map', disparity_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()