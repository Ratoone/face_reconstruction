import logging

import cv2
import numpy as np
import open3d
import scipy.io

from IntrinsicCalibration import IntrinsicCalibration
from StereoCalibration import StereoCalibration


def _grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class ImageProcessing:
    """
    Performs the whole preprocessing pipeline (camera calibration, color normalization) for the given image
    """

    def __init__(self):
        self.camera_left = IntrinsicCalibration("./images/Calibratie 1/calibrationLeft/")
        self.camera_mid = IntrinsicCalibration("./images/Calibratie 1/calibrationMiddle/")
        self.camera_right = IntrinsicCalibration("./images/Calibratie 1/calibrationRight/")

        self.stereo_left = StereoCalibration(self.camera_left, self.camera_mid)
        self.stereo_right = StereoCalibration(self.camera_mid, self.camera_right)

        self.block_matching = cv2.StereoSGBM_create()

        self.logger = logging.getLogger(self.__class__.__name__)

    def calibrate(self):
        self.camera_left.intrinsic_calibration()
        self.camera_mid.intrinsic_calibration()
        self.camera_right.intrinsic_calibration()

        self.stereo_left.calibrate()
        self.stereo_right.calibrate()

    def process_image_batch(self, image_left, image_mid, image_right):
        disparity_left, point_cloud_left = self.process_pair(image_left, image_mid, is_left=True)
        disparity_right, point_cloud_right = self.process_pair(image_mid, image_right, is_left=False)

    def process_pair(self, image_left: np.ndarray, image_right: np.ndarray, is_left: bool = True):
        image_left = cv2.normalize(image_left, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image_right = cv2.normalize(image_right, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        stereo = self.stereo_left if is_left else self.stereo_right
        undistorted_left, undistorted_right = stereo.reproject_images(image_left, image_right)

        undistorted_left = cv2.normalize(undistorted_left, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        undistorted_right = cv2.normalize(undistorted_right, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.logger.info("Computing disparity for {} image pair".format("left" if is_left else "right"))

        cv2.imwrite("undistorted_left.jpg", undistorted_left)
        cv2.imwrite("undistorted_right.jpg", undistorted_right)

        disparity = self.block_matching.compute(undistorted_left, undistorted_right).astype(np.float32) / 16.0
        disparity = (disparity - self.block_matching.getMinDisparity()) / self.block_matching.getNumDisparities()

        point_cloud, mask = self.generate_point_cloud(disparity)
        colored_points = cv2.cvtColor(cv2.normalize(undistorted_left, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), cv2.COLOR_BGR2RGB)
        pcl = open3d.geometry.PointCloud()
        pcl.points = open3d.utility.Vector3dVector(point_cloud)
        pcl.colors = open3d.utility.Vector3dVector(colored_points.reshape(-1, colored_points.shape[-1])[mask])

        return disparity, pcl

    def set_sgbm_parameters(self, num_disparities, min_disparity, block_size, p1, p2, disp_max_dif, uniqueness,
                            speckle_size):
        self.block_matching = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                                    blockSize=block_size,
                                                    P1=p1,
                                                    P2=p2,
                                                    uniquenessRatio=uniqueness,
                                                    minDisparity=min_disparity,
                                                    disp12MaxDiff=disp_max_dif,
                                                    speckleWindowSize=speckle_size,
                                                    speckleRange=1,
                                                    mode=cv2.STEREO_SGBM_MODE_HH
                                                    )

    def generate_point_cloud(self, disparity_image: np.ndarray):
        focal_length = 1530
        Q = np.float32([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, focal_length * 0.05, 0],  # Focal length multiplication obtained experimentally.
                        [0, 0, 0, 1]])

        mask = (disparity_image > disparity_image.min()).reshape(-1)
        point_cloud = cv2.reprojectImageTo3D(disparity_image, Q)
        point_cloud = point_cloud.reshape(-1, point_cloud.shape[-1])
        return point_cloud[mask], mask
