import logging

import cv2
import numpy as np
import open3d

from IntrinsicCalibration import IntrinsicCalibration
from StereoCalibration import StereoCalibration


def _normalize_image(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def _grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class Preprocessing:
    """
    Performs the whole preprocessing pipeline (camera calibration, color normalization) for the given image
    """

    def __init__(self):
        self.calibration_left = IntrinsicCalibration("./images/Calibratie 1/calibrationLeft/")
        self.calibration_mid = IntrinsicCalibration("./images/Calibratie 1/calibrationMiddle/")
        # self.calibration_right = IntrinsicCalibration("./images/Calibratie 1/calibrationRight/")

        self.stereo_left = StereoCalibration(self.calibration_left, self.calibration_mid)
        self.stereo_left.calibrate()
        # self.stereo_right = StereoCalibration(self.calibration_mid, self.calibration_right)
        # self.stereo_right.calibrate()

        self.block_matching = cv2.StereoSGBM().create()

        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess_image_batch(self, image_left: np.ndarray, image_mid: np.ndarray, image_right: np.ndarray):
        image_left = _normalize_image(image_left)
        image_mid = _normalize_image(image_mid)
        image_right = _normalize_image(image_right)

        undistorted_left, undistorted_mid_left = self.stereo_left.reproject_images(image_left, image_mid)
        # undistorted_mid_right, undistorted_right = self.stereo_right.reproject_images(image_mid, image_right)

        self.logger.info("Computing disparity for left and middle image")

        left_disparity = self.block_matching.compute(undistorted_left, undistorted_mid_left)
        left_disparity = cv2.convertScaleAbs(left_disparity, beta=16)

        cv2.imshow("", left_disparity)
        cv2.imshow("left", undistorted_left)
        cv2.imshow("mid left", undistorted_mid_left)
        cv2.waitKey()
        cv2.destroyAllWindows()

        left_point_cloud = cv2.reprojectImageTo3D(left_disparity, self.stereo_left.Q)
        left_point_cloud = left_point_cloud.reshape(-1, left_point_cloud.shape[-1])
        left_point_cloud = left_point_cloud[~np.isinf(left_point_cloud).any(axis=1)]
        pcl = open3d.geometry.PointCloud()
        pcl.points = open3d.utility.Vector3dVector(left_point_cloud)
        open3d.visualization.draw_geometries([pcl])

        # right_disparity = self.block_matching.compute(undistorted_mid_right, undistorted_right)
        # right_disparity = _normalize_image(right_disparity)
