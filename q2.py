import cv2
import numpy as np
import glob

# Load alphabet database
fs = cv2.FileStorage('Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_vertical.txt', cv2.FILE_STORAGE_READ)

# Define 6 letters you want to display
letters = ['O', 'P', 'E', 'N', 'C', 'V']

# Define the board positions (x, y) for each letter
# These are in chessboard coordinates (0-10 for x, 0-7 for y)
letter_positions = [
    (7, 5),  # C position
    (4, 5),  # L position
    (1, 5),  # A position
    (7, 2),  # U position
    (4, 2),  # D position
    (1, 2),  # E position
]

# Load all letter point sets
letter_data = []
for letter in letters:
    charPoints = fs.getNode(letter).mat()
    if charPoints is not None:
        charPoints = charPoints.astype(np.float32).reshape(-1, 3)
        letter_data.append(charPoints)
    else:
        print(f"Warning: Letter '{letter}' not found in database")
        letter_data.append(None)

# Chessboard setup
nx, ny = 11, 8
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in images

images = glob.glob('Dataset_CvDl_Hw1/Q2_Image/*.bmp')

# Calibrate camera
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

# Camera calibration
img_size = cv2.imread(images[0]).shape[:2][::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

# Project all 6 letters onto the chessboard
img = cv2.imread(images[0])

for i, (letter, pos, char_pts) in enumerate(zip(letters, letter_positions, letter_data)):
    if char_pts is None:
        continue
    
    # Translate the letter points to the specified position
    translated_pts = char_pts.copy()
    translated_pts[:, 0] += pos[0]  # x offset
    translated_pts[:, 1] += pos[1]  # y offset
    
    # Project points to image
    projected_pts, _ = cv2.projectPoints(translated_pts, rvecs[0], tvecs[0], mtx, dist)
    
    # Draw the letter with a unique color
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    color = (0, 0, 255)
    
    # Draw points
    for pt in projected_pts:
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(img, (x, y), 3, color, -1)
    
    # Draw lines connecting consecutive points
    for j in range(len(projected_pts) - 1):
        p1 = tuple(np.int32(projected_pts[j].ravel()))
        p2 = tuple(np.int32(projected_pts[j+1].ravel()))
        cv2.line(img, p1, p2, color, 2)

cv2.imshow('Projected Letters', img)
cv2.waitKey(0)
cv2.destroyAllWindows()