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
        self.image_size = None
        self.logger = logging.getLogger(self.__class__.__name__)

        # list of successful calibration photos
        self.successful = np.array([], dtype=bool)

    def intrinsic_calibration(self):
        self.object_points = []  # 3d point in real world space
        self.image_points = []  # 2d points in image plane.

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points
        grid_points = np.zeros((self.inner_x_grid * self.inner_y_grid, 3), np.float32)
        grid_points[:, :2] = np.mgrid[0:self.inner_x_grid, 0:self.inner_y_grid].T.reshape(-1, 2)*100

        for image_name in sorted(glob.glob(os.path.join(self.calibration_image_folder, "*.jpg"))):
            im = cv2.imread(image_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            if self.image_size is None:
                self.image_size = gray.shape[::-1]

            # Find the chess board corners
            found, corners = cv2.findChessboardCorners(gray, (self.inner_x_grid, self.inner_y_grid), None)

            self.successful = np.append(self.successful, found)
            # If found, add object points, image points (after refining them)
            if not found:
                self.logger.info("Calibration failed for image {}".format(image_name))
                self.image_points.append([])
                self.object_points.append([])
                continue

            self.object_points.append(grid_points)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            self.image_points.append(corners2)

        self.image_points = np.array(self.image_points)
        self.object_points = np.array(self.object_points)

        error, self.camera_matrix, self.distortion, _, _ = \
            cv2.calibrateCamera(self.object_points[self.successful], self.image_points[self.successful], self.image_size, None, None, criteria=criteria, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)

        self.logger.info("Intrinsic calibration done with error {}".format(error))
