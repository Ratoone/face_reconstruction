import logging

import cv2

from preprocessing.IntrinsicCalibration import IntrinsicCalibration


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - [%(name)s] %(levelname)s: %(message)s")
    calibration_left = IntrinsicCalibration("./images/Calibratie 1/calibrationMiddle/")
    img = cv2.imread("./images/Calibratie 1/calibrationMiddle/Calibratie 1_M_137.jpg")
    calibration_left.intrinsic_calibration()
    cv2.imshow("original", img)
    cv2.imshow("", calibration_left.undistort_image(img))
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()