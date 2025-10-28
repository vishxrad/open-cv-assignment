import cv2
import numpy as np

def sterio_display(imgL, imgR):
    if imgL is None or imgR is None:
        raise ValueError("imgL/imgR is None")
    if imgL.shape != imgR.shape:
        raise ValueError(f"Image size mismatch: {imgL.shape} vs {imgR.shape}")

    stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)
    disparity = stereo.compute(imgL, imgR)
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return disparity_normalized