import logging

import cv2

from preprocessing.Preprocessing import Preprocessing


def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - [%(name)s] %(levelname)s: %(message)s")
    preprocessing = Preprocessing()

    img1 = cv2.imread("./images/Calibratie 1/calibrationLeft/Calibratie 1_L_137.jpg")
    img2 = cv2.imread("./images/Calibratie 1/calibrationMiddle/Calibratie 1_M_137.jpg")
    img3 = cv2.imread("./images/Calibratie 1/calibrationRight/Calibratie 1_R_137.jpg")

    preprocessing.preprocess_image_batch(img1, img2, img3)


if __name__ == '__main__':
    main()
