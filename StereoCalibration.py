import logging

import cv2
import numpy as np

from IntrinsicCalibration import IntrinsicCalibration


class StereoCalibration:
    """
    Performs the stereo calibration between 2 calibrated cameras. The result will be the rotation and translation
    between the two cameras
    """

    def __init__(self, camera_left: IntrinsicCalibration, camera_right: IntrinsicCalibration):
        self.camera_left = camera_left
        self.camera_right = camera_right
        self.rotation = []
        self.translation = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def calibrate(self):
        mask = np.logical_and(self.camera_left.successful, self.camera_right.successful)
        error, _, _, _, _, self.rotation, self.translation, _, _ = \
            cv2.stereoCalibrate(self.camera_left.object_points[mask],
                                self.camera_left.image_points[mask],
                                self.camera_right.image_points[mask],
                                self.camera_left.camera_matrix,
                                self.camera_left.distortion,
                                self.camera_right.camera_matrix,
                                self.camera_right.distortion,
                                self.camera_left.image_size,
                                flags=cv2.CALIB_FIX_INTRINSIC)

        self.logger.info("Extrinsic calibration done with error {}".format(error))

        self.rotation_left, self.rotation_right, self.perspective_left, self.perspective_right, self.Q, self.roi_left, self.roi_right = \
            cv2.stereoRectify(self.camera_left.camera_matrix,
                              self.camera_left.distortion,
                              self.camera_right.camera_matrix,
                              self.camera_right.distortion,
                              self.camera_left.image_size,
                              self.rotation,
                              self.translation,
                              flags=cv2.CALIB_ZERO_DISPARITY)

    def reproject_images(self, image_left: np.ndarray, image_right: np.ndarray) -> (np.ndarray, np.ndarray):
        x1, y1, w1, h1 = self.roi_left
        x2, y2, w2, h2 = self.roi_right
        y = max(y1, y2)
        h = min(y1 + h1, y2 + h2) - y
        width = image_left.shape[0]
        delta = min(x2 + w2, width - x1)

        map_left1, map_left2 = cv2.initUndistortRectifyMap(self.camera_left.camera_matrix,
                                                           self.camera_left.distortion,
                                                           self.rotation_left,
                                                           self.perspective_left,
                                                           image_left.shape[::-1][1:],
                                                           cv2.CV_32F)

        undistorted_left = cv2.remap(image_left, map_left1, map_left2, cv2.INTER_LINEAR)
        undistorted_left = undistorted_left[y:y+h, width - delta:]

        map_right1, map_right2 = cv2.initUndistortRectifyMap(self.camera_right.camera_matrix,
                                                             self.camera_right.distortion,
                                                             self.rotation_right,
                                                             self.perspective_right,
                                                             image_right.shape[::-1][1:],
                                                             cv2.CV_32F)

        undistorted_right = cv2.remap(image_right, map_right1, map_right2, cv2.INTER_LINEAR)
        undistorted_right = undistorted_right[y:y + h, :delta]

        return cv2.convertScaleAbs(undistorted_left, alpha=255), cv2.convertScaleAbs(undistorted_right, alpha=255)
