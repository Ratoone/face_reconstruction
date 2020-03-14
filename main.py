import logging

import cv2

from preprocessing.IntrinsicCalibration import IntrinsicCalibration
from preprocessing.StereoCalibration import StereoCalibration


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - [%(name)s] %(levelname)s: %(message)s")
    calibration_left = IntrinsicCalibration("./images/Calibratie 1/calibrationLeft/")
    calibration_mid = IntrinsicCalibration("./images/Calibratie 1/calibrationMiddle/")
    calibration_right = IntrinsicCalibration("./images/Calibratie 1/calibrationRight/")

    stereo_left = StereoCalibration()
    stereo_left.calibrate(calibration_mid, calibration_left)
    stereo_right = StereoCalibration()
    stereo_right.calibrate(calibration_mid, calibration_right)

    img = cv2.imread("./images/Calibratie 1/calibrationMiddle/Calibratie 1_M_137.jpg")
    cv2.imshow("original", img)
    cv2.imshow("", calibration_left.undistort_image(img))
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()