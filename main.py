import logging
import sys

import cv2
import numpy as np
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider

from ImageProcessing import ImageProcessing


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("pipeline.ui", self)

        self.show()

        self.process = ImageProcessing()
        self.img1 = cv2.imread("./images/subject1/subject1Left/subject1_Left_1.jpg")
        # self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.imread("./images/subject1/subject1Middle/subject1_Middle_1.jpg")
        # self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.img3 = cv2.imread("./images/subject1/subject1Right/subject1_Right_1.jpg")

        self.compute_disparity_button.clicked.connect(self.recompute_disparity)
        self.calibrate_button.clicked.connect(self.calibrate)
        self.show_pcl_button.clicked.connect(self.show_pcl)

        self.image_placeholder = self.findChild(QLabel, "disparity_image")

        self.block_size_slider = self.findChild(QSlider, "block_size")
        self.block_size_slider.valueChanged.connect(self._validate_values)
        self.min_disparity_slider = self.findChild(QSlider, "min_disparity")
        self.min_disparity_slider.valueChanged.connect(self._show_values)
        self.num_disparity_slider = self.findChild(QSlider, "num_disparity")
        self.num_disparity_slider.valueChanged.connect(self._show_values)
        self.p1_slider = self.findChild(QSlider, "p1")
        self.p1_slider.valueChanged.connect(self._validate_values)
        self.p2_slider = self.findChild(QSlider, "p2")
        self.p2_slider.valueChanged.connect(self._validate_values)
        self.max_dif_slider = self.findChild(QSlider, "disp_max_dif")
        self.max_dif_slider.valueChanged.connect(self._show_values)
        self.uniqueness_slider = self.findChild(QSlider, "uniqueness")
        self.uniqueness_slider.valueChanged.connect(self._show_values)
        self.speckle_slider = self.findChild(QSlider, "speckle_size")
        self.speckle_slider.valueChanged.connect(self._show_values)

    def _validate_values(self):
        if self.block_size_slider.value() % 2 == 0:
            self.block_size_slider.setValue(self.block_size_slider.value() - 1)

        if self.p1_slider.value() >= self.p2_slider.value():
            self.p2_slider.setValue(self.p1_slider.value() + 1)

        self.p1_slider.setValue(8*self.block_size_slider.value()**2)
        self.p2_slider.setValue(32*self.block_size_slider.value()**2)

        self._show_values()

    def _show_values(self):
        self.block_size_value.setText(str(self.block_size_slider.value()))
        self.min_disparity_value.setText(str(self.min_disparity_slider.value()))
        self.num_disparity_value.setText(str(self.num_disparity_slider.value()*16))
        self.p1_value.setText(str(self.p1_slider.value()))
        self.p2_value.setText(str(self.p2_slider.value()))
        self.max_diff_value.setText(str(self.max_dif_slider.value()))
        self.uniqueness_value.setText(str(self.uniqueness_slider.value()))
        self.speckle_value.setText(str(self.speckle_slider.value()))

    def calibrate(self):
        self.calibrate_button.setEnabled(False)
        self.process.calibrate()
        self.show_image(self.process.preprocess_image_batch(self.img1, self.img2))

    def recompute_disparity(self):
        self.process.set_sgbm_parameters(self.num_disparity_slider.value() * 16,
                                         self.min_disparity_slider.value(),
                                         self.block_size_slider.value(),
                                         self.p1_slider.value(),
                                         self.p2_slider.value(),
                                         self.max_dif_slider.value(),
                                         self.uniqueness_slider.value(),
                                         self.speckle_slider.value()
                                         )
        self.show_image(self.process.preprocess_image_batch(self.img1, self.img2))

    def show_pcl(self):
        img = self.process.preprocess_image_batch(self.img1, self.img2)
        self.process.generate_point_cloud(img)

    def show_image(self, image: np.ndarray):
        cv2.imshow("", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

        qimg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Indexed8)
        self.image_placeholder.setGeometry(QtCore.QRect(self.image_placeholder.x(), self.image_placeholder.y(), image.shape[1], image.shape[0]))
        self.image_placeholder.setPixmap(QPixmap.fromImage(qimg))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] %(levelname)s: %(message)s")
    app = QApplication(sys.argv)

    window = MainWindow()

    app.exec_()
