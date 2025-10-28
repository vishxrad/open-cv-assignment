import sys
import os
import glob
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QSpinBox, QFileDialog, QDialog,
    QTextEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

# Assuming q3.py exists in the same directory
try:
    from q3 import sterio_display
except ImportError:
    print("Warning: q3.py not found. '3.1 stereo disparity map' will not work.")
    # Create a dummy function so the program doesn't crash
    def sterio_display(imgL, imgR):
        print("Error: sterio_display function not loaded.")
        # Return a black image as a placeholder
        return np.zeros(imgL.shape, dtype=np.uint8)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CvDL_Hw")
        self.setGeometry(100, 100, 1000, 750)  # Adjusted size for new layout

        # Initialize variable to store folder path
        self.folder_path = None
        self.ar_folder_path = None
        
        # Initialize variables to store stereo image paths
        self.stereo_left_path = None
        self.stereo_right_path = None

        # Calibration variables
        self.objpoints = []
        self.imgpoints = []
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.gray_shape = None
        self.image_files = []

        # SIFT variables
        self.sift_image1_path = None
        self.sift_image2_path = None
        self.sift_image1 = None
        self.sift_image2 = None

        # Create a central widget and a main horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Column Layout ---
        left_column_layout = QVBoxLayout()

        # 1. Calibration Group
        calib_group = QGroupBox("1.Calibration")
        calib_vbox = QVBoxLayout()

        self.btn_load_folder = QPushButton("Load folder")
        self.btn_load_folder.clicked.connect(self.load_folder)
        calib_vbox.addWidget(self.btn_load_folder)

        self.btn_find_corners = QPushButton("1.1 Find Corners")
        self.btn_find_corners.clicked.connect(self.find_corners)
        self.btn_find_intrinsic = QPushButton("1.2 Find Intrinsic")
        self.btn_find_intrinsic.clicked.connect(self.find_intrinsic)

        self.label_intrinsic_matrix = QTextEdit()
        self.label_intrinsic_matrix.setPlaceholderText("Intrinsic Matrix (K)")
        self.label_intrinsic_matrix.setReadOnly(True)

        self.label_extrinsic = QLabel("Select image for Extrinsic:") # Sub-label
        self.spinbox_extrinsic = QSpinBox()
        self.btn_find_extrinsic = QPushButton("1.3 Find extrinsic")
        self.btn_find_extrinsic.clicked.connect(self.find_extrinsic)
        
        self.label_extrinsic_matrix = QTextEdit()
        self.label_extrinsic_matrix.setPlaceholderText("Extrinsic Matrix")
        self.label_extrinsic_matrix.setReadOnly(True)

        self.btn_find_distortion = QPushButton("1.4 Find distortion")
        self.btn_find_distortion.clicked.connect(self.find_distortion)

        self.label_distortion_matrix = QTextEdit()
        self.label_distortion_matrix.setPlaceholderText("Distortion Matrix")
        self.label_distortion_matrix.setReadOnly(True)

        self.btn_show_undistortion = QPushButton("1.5 Show undistortion")
        self.btn_show_undistortion.clicked.connect(self.show_undistortion)

        calib_vbox.addWidget(self.btn_find_corners)
        calib_vbox.addWidget(self.btn_find_intrinsic)
        calib_vbox.addWidget(self.label_intrinsic_matrix)
        calib_vbox.addWidget(self.label_extrinsic)
        calib_vbox.addWidget(self.spinbox_extrinsic)
        calib_vbox.addWidget(self.btn_find_extrinsic)
        calib_vbox.addWidget(self.label_extrinsic_matrix)
        calib_vbox.addWidget(self.btn_find_distortion)
        calib_vbox.addWidget(self.label_distortion_matrix)
        calib_vbox.addWidget(self.btn_show_undistortion)
        calib_group.setLayout(calib_vbox)
        left_column_layout.addWidget(calib_group)
        
        right_column_layout = QVBoxLayout()

        # 2. Augmented Reality Group
        ar_group = QGroupBox("2.Augmented Reality")
        ar_vbox = QVBoxLayout()
        self.btn_ar_load_folder = QPushButton("Load folder")
        self.btn_ar_load_folder.clicked.connect(self.load_ar_folder)
        self.le_ar_text = QLineEdit()
        self.le_ar_text.setPlaceholderText("Enter text...")
        self.btn_ar_board = QPushButton("2.1 show words on board")
        self.btn_ar_board.clicked.connect(self.show_words_on_board)
        self.btn_ar_vertical = QPushButton("2.2 show words vertical")
        self.btn_ar_vertical.clicked.connect(self.show_words_vertical)

        ar_vbox.addWidget(self.btn_ar_load_folder)
        ar_vbox.addWidget(self.le_ar_text)
        ar_vbox.addWidget(self.btn_ar_board)
        ar_vbox.addWidget(self.btn_ar_vertical)
        ar_group.setLayout(ar_vbox)
        right_column_layout.addWidget(ar_group)

        # 3. Stereo Disparity Map
        stereo_group = QGroupBox("3.Stereo Disparity Map")
        stereo_vbox = QVBoxLayout()

        self.btn_stereo_load_left = QPushButton("Load Image Left")
        self.btn_stereo_load_left.clicked.connect(self.load_stereo_left)

        self.btn_stereo_load_right = QPushButton("Load Image Right")
        self.btn_stereo_load_right.clicked.connect(self.load_stereo_right)

        self.btn_stereo_map = QPushButton("3.1 stereo disparity map")
        self.btn_stereo_map.clicked.connect(self.show_stereo_map)
        stereo_vbox.addWidget(self.btn_stereo_load_left)
        stereo_vbox.addWidget(self.btn_stereo_load_right)
        stereo_vbox.addWidget(self.btn_stereo_map)
        stereo_group.setLayout(stereo_vbox)
        right_column_layout.addWidget(stereo_group)

        # 4. SIFT Group
        sift_group = QGroupBox("4.SIFT")
        sift_vbox = QVBoxLayout()
        self.btn_sift_load1 = QPushButton("Load Image1")
        self.btn_sift_load1.clicked.connect(self.load_sift_image1)
        self.btn_sift_load2 = QPushButton("Load Image2")
        self.btn_sift_load2.clicked.connect(self.load_sift_image2)
        self.btn_keypoints = QPushButton("4.1 Keypoints")
        self.btn_keypoints.clicked.connect(self.show_sift_keypoints)
        self.btn_matched_keypoints = QPushButton("4.2 Matched Keypoints")
        self.btn_matched_keypoints.clicked.connect(self.show_matched_keypoints)

        sift_vbox.addWidget(self.btn_sift_load1)
        sift_vbox.addWidget(self.btn_sift_load2)
        sift_vbox.addWidget(self.btn_keypoints)
        sift_vbox.addWidget(self.btn_matched_keypoints)
        sift_group.setLayout(sift_vbox)
        right_column_layout.addWidget(sift_group)
        

        # --- Add Columns to Main Layout ---
        # Add layouts with a stretch factor of 1 to allow them to share space
        main_layout.addLayout(left_column_layout, 1)
        main_layout.addLayout(right_column_layout, 1)


    def _show_image(self, title, img):
        """A robust method to display both grayscale and BGR images with proper scaling."""
        if img is None:
            print(f"Error: Image for '{title}' is None.")
            return

        # Determine image format and create QImage
        if len(img.shape) == 2:  # Grayscale
            h, w = img.shape
            # Ensure data is contiguous
            img_cont = np.ascontiguousarray(img)
            qimg = QImage(img_cont.data, w, h, w, QImage.Format_Grayscale8)
        elif len(img.shape) == 3:  # BGR
            # Convert BGR (OpenCV default) to RGB for QImage
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             # Ensure data is contiguous
            rgb_image_cont = np.ascontiguousarray(rgb_image)
            h, w, ch = rgb_image_cont.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image_cont.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            print("Error: Unsupported image format.")
            return

        pixmap = QPixmap.fromImage(qimg)

        # Create a dialog and a label to display the image
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        lbl = QLabel(dlg)
        
        # Get screen size to scale the image appropriately
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_w, screen_h = screen_geometry.width(), screen_geometry.height()

        # Scale pixmap if it's larger than the screen, maintaining aspect ratio
        if pixmap.width() > screen_w * 0.9 or pixmap.height() > screen_h * 0.9:
            pixmap = pixmap.scaled(int(screen_w * 0.9), int(screen_h * 0.9), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        lbl.setPixmap(pixmap)
        
        # The layout will manage the label's size
        layout = QVBoxLayout(dlg)
        layout.addWidget(lbl)
        dlg.setLayout(layout)
        
        # Resize the dialog to fit the pixmap content
        dlg.adjustSize()
        dlg.exec_()

    def load_folder(self):
        """Open a dialog to select a folder and save the path"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            "",  # Start directory (empty means default)
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:  # If user didn't cancel
            self.folder_path = folder_path
            print(f"Selected folder: {self.folder_path}")  # For debugging
            self.image_files = sorted(glob.glob(os.path.join(self.folder_path, '*.bmp')))
            if self.image_files:
                self.spinbox_extrinsic.setMinimum(1)
                self.spinbox_extrinsic.setMaximum(len(self.image_files))
            else:
                print("No .bmp images found in the selected folder.")
        else:
            print("No folder selected")
    
    def load_ar_folder(self):
        """Open a dialog to select a folder for Augmented Reality"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select AR Image Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.ar_folder_path = folder_path
            print(f"Selected AR folder: {self.ar_folder_path}")
        else:
            print("No AR folder selected")

    def load_stereo_left(self):
        """Open a dialog to select the left stereo image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Left Stereo Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.stereo_left_path = file_path
            print(f"Selected left image: {self.stereo_left_path}")
        else:
            print("No left image selected")
    
    def load_stereo_right(self):
        """Open a dialog to select the right stereo image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Right Stereo Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.stereo_right_path = file_path
            print(f"Selected right image: {self.stereo_right_path}")
        else:
            print("No right image selected")

    def load_sift_image1(self):
        """Open a dialog to select the first image for SIFT."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 1 for SIFT", "", "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if file_path:
            self.sift_image1_path = file_path
            self.sift_image1 = cv2.imread(file_path)
            print(f"Loaded SIFT Image 1: {file_path}")
        else:
            print("No image selected for SIFT 1.")

    def load_sift_image2(self):
        """Open a dialog to select the second image for SIFT."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image 2 for SIFT", "", "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if file_path:
            self.sift_image2_path = file_path
            self.sift_image2 = cv2.imread(file_path)
            print(f"Loaded SIFT Image 2: {file_path}")
        else:
            print("No image selected for SIFT 2.")

    def _perform_ar_calibration(self):
        """Internal method to run calibration for the AR images."""
        if not self.ar_folder_path:
            print("Please load an AR folder first.")
            return None, None, None, None

        objpoints = []
        imgpoints = []
        nx, ny = 11, 8
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        ar_images = sorted(glob.glob(os.path.join(self.ar_folder_path, '*.bmp')))
        if not ar_images:
            print("No .bmp images found in the AR folder.")
            return None, None, None, None

        gray_shape = None
        for fname in ar_images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray_shape is None:
                gray_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)

        if not objpoints:
            print("Could not find chessboard corners in any AR images.")
            return None, None, None, None

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
        return mtx, dist, rvecs, tvecs

    def _project_and_draw_text(self, db_path):
        """Helper function to project text from a given database onto the board."""
        text = self.le_ar_text.text().upper()
        if not text:
            print("Please enter text to display.")
            return

        if not self.ar_folder_path:
            print("Please load an AR folder first.")
            return

        mtx, dist, rvecs, tvecs = self._perform_ar_calibration()
        if mtx is None:
            print("AR calibration failed.")
            return

        # Load alphabet database
        # Try to find the database relative to the script or in a common location
        script_dir = os.path.dirname(os.path.realpath(__file__))
        
        # Check for db_path as an absolute path or relative to script
        full_db_path = db_path
        if not os.path.exists(full_db_path):
            full_db_path = os.path.join(script_dir, db_path)
            if not os.path.exists(full_db_path):
                 # As a last resort, check relative to current working directory
                 full_db_path = os.path.abspath(db_path)
                 if not os.path.exists(full_db_path):
                    print(f"Error: Could not find alphabet database at {db_path} or related paths.")
                    return

        fs = cv2.FileStorage(full_db_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"Error: Could not open alphabet database at {full_db_path}")
            return

        # Define the fixed board positions for each letter
        letter_positions = [
            (7, 5), (4, 5), (1, 5),
            (7, 2), (4, 2), (1, 2),
        ]

        ar_images = sorted(glob.glob(os.path.join(self.ar_folder_path, '*.bmp')))
        img = cv2.imread(ar_images[0])

        # Iterate through the input text and the fixed positions
        # This will only draw up to the number of available positions (6)
        for i, char in enumerate(text):
            if i >= len(letter_positions):
                print(f"Warning: Input text is longer than the number of available positions. Truncating to {len(letter_positions)} letters.")
                break

            node = fs.getNode(char)
            if node.empty():
                print(f"Warning: Letter '{char}' not found in database.")
                continue

            char_pts = node.mat().astype(np.float32).reshape(-1, 3)
            pos = letter_positions[i] # Get the position for the current letter

            # Translate points to the desired position on the board
            translated_pts = char_pts.copy()
            translated_pts[:, 0] += pos[0]  # x offset
            translated_pts[:, 1] += pos[1]  # y offset

            projected_pts, _ = cv2.projectPoints(translated_pts, rvecs[0], tvecs[0], mtx, dist)

            # Draw lines connecting consecutive points
            for j in range(0, len(projected_pts) - 1, 2):
                p1 = tuple(np.int32(projected_pts[j].ravel()))
                p2 = tuple(np.int32(projected_pts[j+1].ravel()))
                cv2.line(img, p1, p2, (0, 0, 255), 5) # Thicker line
        
        fs.release()
        self._show_image(f"AR: '{text}'", img)

    def show_words_on_board(self):
        """Projects 2D text onto the chessboard."""
        # Assuming a database file exists in a known relative path
        db_path = 'Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_onboard.txt'
        self._project_and_draw_text(db_path)

    def show_words_vertical(self):
        """Projects 3D/vertical text onto the chessboard."""
        db_path = 'Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_vertical.txt'
        self._project_and_draw_text(db_path)

    def _perform_calibration(self):
        """Internal method to run the calibration process on the loaded folder."""
        if not self.folder_path:
            print("Please load a folder first.")
            return False
        
        # Clear previous results
        self.objpoints = []
        self.imgpoints = []

        # Chessboard pattern
        nx, ny = 11, 8
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        print(f"Found {len(self.image_files)} images for calibration.")
        for fname in self.image_files:
            image = cv2.imread(fname)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.gray_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
            
            if ret:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.imgpoints.append(corners2)
        
        if not self.objpoints or not self.imgpoints:
            print("Could not find chessboard corners in any of the images.")
            return False

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.gray_shape, None, None
        )
        return True

    def find_corners(self):
        """Find and display chessboard corners for the first image in the folder."""
        if not self.image_files:
            print("Please load a folder with images first.")
            return

        fname = self.image_files[0]
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        nx, ny = 11, 8
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
        
        if ret:
            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            self._show_image('Chessboard Corners', image)
        else:
            print("Could not find corners in the first image.")

    def find_intrinsic(self):
        """Calculate and display the intrinsic matrix."""
        if self._perform_calibration():
            self.label_intrinsic_matrix.setText(np.array2string(self.mtx, precision=4))
            print("Intrinsic matrix calculated and displayed.")
        else:
            print("Calibration failed. Could not find intrinsic matrix.")

    def find_extrinsic(self):
        """Find and display the extrinsic matrix for the selected image."""
        if self.mtx is None or not self.rvecs or not self.tvecs:
            print("Please find the intrinsic matrix first (run 1.2).")
            return
        
        img_idx = self.spinbox_extrinsic.value() - 1
        if 0 <= img_idx < len(self.rvecs):
            rvec = self.rvecs[img_idx]
            tvec = self.tvecs[img_idx]
            
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            extrinsic_matrix = np.hstack((rotation_matrix, tvec))
            
            self.label_extrinsic_matrix.setText(np.array2string(extrinsic_matrix, precision=4))
            print(f"Displaying extrinsic matrix for image {img_idx + 1}.")
        else:
            print("Invalid image index selected.")

    def find_distortion(self):
        """Display the distortion matrix."""
        if self.dist is None:
            print("Please find the intrinsic matrix first (run 1.2) to get distortion.")
            return
        self.label_distortion_matrix.setText(np.array2string(self.dist, precision=4))
        print("Distortion matrix displayed.")

    def show_undistortion(self):
        """Show the original and undistorted version of the selected image side-by-side."""
        if self.mtx is None or self.dist is None:
            print("Please find the intrinsic matrix first (run 1.2).")
            return
            
        img_idx = self.spinbox_extrinsic.value() - 1
        if 0 <= img_idx < len(self.image_files):
            fname = self.image_files[img_idx]
            original_img = cv2.imread(fname)
            
            undistorted_img = cv2.undistort(original_img, self.mtx, self.dist, None, self.mtx)
            
            # Resize for consistent display
            h, w, _ = original_img.shape
            combined_img = np.hstack((original_img, undistorted_img))
            
            self._show_image(f"Original vs Undistorted (Image {img_idx + 1})", combined_img)
        else:
            print("Invalid image index selected.")

    def show_sift_keypoints(self):
        """Detect and display SIFT keypoints on the first loaded image."""
        if self.sift_image1 is None:
            print("Please load Image 1 for SIFT first.")
            return

        gray = cv2.cvtColor(self.sift_image1, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(gray, None)
        
        img_with_keypoints = cv2.drawKeypoints(self.sift_image1, keypoints, None, color=(0, 255, 0))
        
        self._show_image("SIFT Keypoints", img_with_keypoints)

    def show_matched_keypoints(self):
        """Detect, match, and display SIFT keypoints between two images."""
        if self.sift_image1 is None or self.sift_image2 is None:
            print("Please load both Image 1 and Image 2 for SIFT.")
            return

        gray1 = cv2.cvtColor(self.sift_image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.sift_image2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        if descriptors1 is None or descriptors2 is None:
            print("Could not compute descriptors for one or both SIFT images.")
            return

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        matched_img = cv2.drawMatchesKnn(
            self.sift_image1, keypoints1, 
            self.sift_image2, keypoints2, 
            good_matches, None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        self._show_image("Matched SIFT Keypoints", matched_img)

    def show_stereo_map(self):
        """Display the stereo disparity map"""
        if not (self.stereo_left_path and self.stereo_right_path):
            print("Please load both left and right images first")
            return

        imgL = cv2.imread(self.stereo_left_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(self.stereo_right_path, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            print("Error: failed to read one or both images.")
            return
        if imgL.shape != imgR.shape:
            print(f"Error: image size mismatch {imgL.shape} vs {imgR.shape}")
            return

        disp = sterio_display(imgL, imgR)
        self._show_image("Disparity Map", disp)

    def _show_gray_image(self, title, img):
        self._show_image(title, img)

    def _show_bgr_image(self, title, img):
        self._show_image(title, img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Apply a light, professional stylesheet with BLACK buttons
    app.setStyleSheet("""
        QMainWindow {
            background-color: #F0F2F5; /* A very light, neutral background */
        }
        QGroupBox {
            background-color: #FFFFFF; /* White for group boxes */
            border: 1px solid #DCDCDC; /* Light gray border */
            border-radius: 8px;
            margin-top: 10px; /* Provides space for the title */
            font-family: "Segoe UI", sans-serif;
            font-size: 15px;
            font-weight: bold;
            color: #005A9C; /* A deep professional blue for the title */
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 5px 10px;
            background-color: #FFFFFF;
            border-top-left-radius: 8px; /* Match groupbox radius */
            border-top-right-radius: 8px; /* Match groupbox radius */
            border-bottom: 1px solid #DCDCDC; /* Separator line */
        }
        
        /* --- MODIFIED BUTTON STYLE --- */
        QPushButton {
            background-color: #000000; /* Plain black */
            color: #FFFFFF; /* White text */
            border: 1px solid #333333; /* Dark border for definition */
            border-radius: 5px;
            padding: 8px 12px;
            font-size: 13px;
            font-weight: bold;
            min-height: 25px;
        }
        QPushButton:hover {
            background-color: #222222; /* Very dark gray on hover */
            border: 1px solid #555555;
        }
        QPushButton:pressed {
            background-color: #111111; /* Even darker gray when pressed */
        }
        /* ----------------------------- */
        
        QLabel {
            color: #333333; /* Standard dark text for labels */
            font-size: 13px;
            padding-left: 2px; /* Small padding for alignment */
        }
        QLineEdit, QTextEdit, QSpinBox {
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #CCCCCC; /* Standard light gray border */
            border-radius: 4px;
            padding: 5px;
            font-size: 13px;
            /* Set a minimum height to help with spacing */
            min-height: 25px; 
        }
        QTextEdit {
             /* Allow text edits to grow more */
            min-height: 60px;
        }
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {
            border: 1px solid #0078D4; /* Blue border on focus */
        }
        QTextEdit[readOnly="true"] {
             background-color: #F0F2F5; /* Match window bg for read-only */
        }
        QSpinBox::up-button, QSpinBox::down-button {
            subcontrol-origin: border;
            width: 16px;
            border-width: 1px;
            border-style: solid;
            border-color: #CCCCCC;
            background-color: #F0F2F5;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {
            background-color: #E0E0E0;
        }
        QSpinBox::up-arrow {
            image: url(icons/up_arrow.png); /* For a custom icon */
            width: 9px;
            height: 9px;
        }
        QSpinBox::down-arrow {
            image: url(icons/down_arrow.png); /* For a custom icon */
            width: 9px;
            height: 9px;
        }
        QDialog {
            background-color: #FFFFFF;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())