import glob
import logging
import os

import numpy as np
import cv2


class IntrinsicCalibration:
    """
    Responsible for intrinsic camera calibration
    """

    def __init__(self, calibration_image_folder: str):
        self.calibration_image_folder = calibration_image_folder
        # The grid in this example is 7x10 squares, but we need the number of internal points
        self.inner_x_grid = 6
        self.inner_y_grid = 9
        self.camera_matrix = []
        self.logger = logging.getLogger(self.__class__.__name__)

        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

    def intrinsic_calibration(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points
        grid_points = np.zeros((self.inner_x_grid * self.inner_y_grid, 3), np.float32)
        grid_points[:, :2] = np.mgrid[0:self.inner_x_grid, 0:self.inner_y_grid].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        gray = []

        for image_name in glob.glob(os.path.join(self.calibration_image_folder, "*.jpg")):
            im = cv2.imread(image_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            found, corners = cv2.findChessboardCorners(gray, (self.inner_x_grid, self.inner_y_grid), None)

            # If found, add object points, image points (after refining them)
            if not found:
                self.logger.info("Calibration not found for {}".format(image_name))
                continue

            self.objpoints.append(grid_points)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.imgpoints.append(corners2)

        _, self.camera_matrix, self.distortion, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        self.camera_matrix_augmented, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion, gray.shape[::-1], 0)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.camera_matrix, self.distortion, None, self.camera_matrix_augmented)
