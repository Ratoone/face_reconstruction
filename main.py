import logging
import sys

import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QSlider
from PyQt5 import uic

from ImageProcessing import Preprocessing


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("pipeline.ui", self)

        self.show()

        self.compute_disparity = self.findChild(QPushButton, "compute_disparity")
        self.compute_disparity.clicked.connect(lambda: print("shit"))

        self.image_placeholder = self.findChild(QLabel, "disparity_image")

        self.block_size_slider = self.findChild(QSlider, "block_size")
        self.min_disparity_slider = self.findChild(QSlider, "min_disparity")
        self.num_disparity_slider = self.findChild(QSlider, "num_disparity")
        self.p1_slider = self.findChild(QSlider, "p1")
        self.p2_slider = self.findChild(QSlider, "p2")
        self.max_dif_slider = self.findChild(QSlider, "disp_max_dif")
        self.uniqueness_slider = self.findChild(QSlider, "uniqueness")


def main():
    preprocessing = Preprocessing()

    img1 = cv2.imread("./images/subject1/subject1Left/subject1_Left_1.jpg")
    img2 = cv2.imread("./images/subject1/subject1Middle/subject1_Middle_1.jpg")
    img3 = cv2.imread("./images/subject1/subject1Right/subject1_Right_1.jpg")

    # img1 = cv2.imread("./images/Calibratie 1/calibrationLeft/Calibratie 1_L_1.jpg")
    # img2 = cv2.imread("./images/Calibratie 1/calibrationMiddle/Calibratie 1_M_1.jpg")
    # img3 = cv2.imread("./images/subject1/subject1Right/subject1_Right_1.jpg")

    preprocessing.preprocess_image_batch(img1, img2, img3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] %(levelname)s: %(message)s")
    app = QApplication(sys.argv)

    window = MainWindow()

    app.exec_()