import cv2
import numpy as np
import glob


nx, ny = 11, 8

objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in images



images = glob.glob('Dataset_CvDl_Hw1/Q1_Image/*.bmp')

for fname in images:
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
    
    if ret:
        # Refine corners
        corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), 
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Append to lists
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and show corners (optional)
        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
        cv2.imshow('Corners', image)
        cv2.waitKey(100) 


#1.2
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Intrinsic matrix (K):\n", mtx)

#1.3
rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
extrinsic_matrix = np.hstack((rotation_matrix , tvecs[0]))

print("Extrensic Matrix: ",extrinsic_matrix)

#1.4
print("Distortion Matrix: ",dist)

#1.5
result_img = cv2.undistort(gray, mtx, dist)
cv2.imshow('Corners', result_img)
cv2.waitKey(1000)