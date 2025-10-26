import cv2
import numpy as np

# Load images
left = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Left.jpg')
right = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Right.jpg')

# Check if images loaded correctly
if left is None or right is None:
    print("Error: Could not read one or both images.")
    exit()

grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# Create one SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypointsL, descriptorsL = sift.detectAndCompute(grayL, None)
keypointsR, descriptorsR = sift.detectAndCompute(grayR, None)

# Match descriptors using BFMatcher with k-Nearest Neighbors
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptorsL, descriptorsR, k=2)

# Apply Lowe's ratio test to find good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

# Draw the good matches
# The output window title is changed to be more descriptive
image = cv2.drawMatchesKnn(grayL, keypointsL, grayR, keypointsR, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Good Keypoint Matches', image)
cv2.waitKey(10000)
cv2.destroyAllWindows()