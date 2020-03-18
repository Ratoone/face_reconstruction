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


class ImageProcessing:
    """
    Performs the whole preprocessing pipeline (camera calibration, color normalization) for the given image
    """

    def __init__(self):
        self.calibration_left = IntrinsicCalibration("./images/Calibratie 1/calibrationLeft/")
        self.calibration_mid = IntrinsicCalibration("./images/Calibratie 1/calibrationMiddle/")
        self.calibration_right = IntrinsicCalibration("./images/Calibratie 1/calibrationRight/")

        self.stereo_left = StereoCalibration(self.calibration_left, self.calibration_mid)
        # self.stereo_right = StereoCalibration(self.calibration_mid, self.calibration_right)

        self.block_matching = cv2.StereoSGBM().create()

        self.logger = logging.getLogger(self.__class__.__name__)

    def calibrate(self):
        self.calibration_left.intrinsic_calibration()
        self.calibration_mid.intrinsic_calibration()
        self.calibration_right.intrinsic_calibration()

        self.stereo_left.calibrate()
        # self.stereo_right.calibrate()

    def preprocess_image_batch(self, image_left: np.ndarray, image_right: np.ndarray, is_left: bool = True):
        image_left = _normalize_image(image_left)
        image_right = _normalize_image(image_right)

        stereo = self.stereo_left if is_left else self.stereo_right
        undistorted_left, undistorted_right = stereo.reproject_images(image_left, image_right)

        self.logger.info("Computing disparity for {} image pair".format("left" if is_left else "right"))

        cv2.imwrite("undistorted_left.jpg", undistorted_left)
        cv2.imwrite("undistorted_right.jpg", undistorted_right)

        disparity = self.block_matching.compute(undistorted_left, undistorted_right)
        alpha = 1.0
        disparity = cv2.convertScaleAbs(disparity, alpha=alpha, beta=16*alpha)
        cv2.imwrite("disparity.jpg", disparity)

        return disparity

    def set_sgbm_parameters(self, num_disparities, min_disparity, block_size, p1, p2, disp_max_dif, uniqueness, speckle_size):
        self.block_matching = cv2.StereoSGBM().create(numDisparities=num_disparities,
                                                      blockSize=block_size,
                                                      P1=p1,
                                                      P2=p2,
                                                      uniquenessRatio=uniqueness,
                                                      minDisparity=min_disparity,
                                                      disp12MaxDiff=disp_max_dif,
                                                      speckleWindowSize=speckle_size,
                                                      speckleRange=1
                                                      )
        # self.block_matching = cv2.StereoBM().create(blockSize=block_size, numDisparities=num_disparities)

    def generate_point_cloud(self, disparity_image: np.ndarray, is_left: bool = True):
        if is_left:
            Q = self.stereo_left.Q
        else:
            Q = self.stereo_right.Q

        point_cloud = cv2.reprojectImageTo3D(disparity_image, Q)
        point_cloud = point_cloud.reshape(-1, point_cloud.shape[-1])
        point_cloud = point_cloud[~np.isinf(point_cloud).any(axis=1)]

        pcl = open3d.geometry.PointCloud()
        pcl.points = open3d.utility.Vector3dVector(point_cloud)
        open3d.visualization.draw_geometries([pcl])
