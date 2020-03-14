import logging

import cv2
import numpy as np

from preprocessing.IntrinsicCalibration import IntrinsicCalibration


class StereoCalibration:
    """
    Performs the stereo calibration between 2 calibrated images. The result will be the rotation and translation
    between the two images
    """

    def __init__(self):
        self.rotation = []
        self.translation = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def calibrate(self, reference_calibration: IntrinsicCalibration, target_calibration: IntrinsicCalibration):
        mask = np.logical_and(reference_calibration.successful, target_calibration.successful)
        error, _, _, _, _, self.rotation, self.translation, _, _ =\
            cv2.stereoCalibrate(reference_calibration.object_points[mask],
                                reference_calibration.image_points[mask],
                                target_calibration.image_points[mask],
                                reference_calibration.camera_matrix,
                                reference_calibration.distortion,
                                target_calibration.camera_matrix,
                                target_calibration.distortion,
                                reference_calibration.image_size,
                                flags=cv2.CALIB_FIX_INTRINSIC)

        self.logger.debug("Extrinsic calibration done with error {}".format(error))
