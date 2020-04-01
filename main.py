import logging
import sys
from functools import partial

import cv2
import numpy as np
import open3d
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider

from ImageProcessing import ImageProcessing
from PointCloudProcessing import PointCloudProcessing


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("pipeline.ui", self)

        self.show()

        self.process = ImageProcessing()
        self.point_cloud_processing = PointCloudProcessing()
        self.img1 = cv2.imread("./images/subject1/subject1Left/subject1_Left_1.jpg")
        self.img2 = cv2.imread("./images/subject1/subject1Middle/subject1_Middle_1.jpg")
        self.img3 = cv2.imread("./images/subject1/subject1Right/subject1_Right_1.jpg")

        self.compute_disparity_left_button.clicked.connect(partial(self.recompute_disparity, True))
        self.compute_disparity_right_button.clicked.connect(partial(self.recompute_disparity, False))
        self.calibrate_button.clicked.connect(self.calibrate)
        self.process_button.clicked.connect(self.process_pcl)

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

        self._validate_values()

    def _validate_values(self):
        if self.block_size_slider.value() % 2 == 0:
            self.block_size_slider.setValue(self.block_size_slider.value() - 1)

        if self.p1_slider.value() >= self.p2_slider.value():
            self.p2_slider.setValue(self.p1_slider.value() + 1)

        self.p1_slider.setValue(8*3*self.block_size_slider.value()**2)
        self.p2_slider.setValue(32*3*self.block_size_slider.value()**2)

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

    def recompute_disparity(self, is_left):
        self.process.set_sgbm_parameters(self.num_disparity_slider.value() * 16,
                                         self.min_disparity_slider.value(),
                                         self.block_size_slider.value(),
                                         self.p1_slider.value(),
                                         self.p2_slider.value(),
                                         self.max_dif_slider.value(),
                                         self.uniqueness_slider.value(),
                                         self.speckle_slider.value()
                                         )
        if is_left:
            image_left = self.img1
            image_right = self.img2
        else:
            image_left = self.img2
            image_right = self.img3

        disparity, pcl = self.process.process_pair(image_left, image_right, is_left=is_left)
        self.show_image(disparity)
        cv2.imwrite("disparity_{}.jpg".format("left" if is_left else "right"), cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        open3d.visualization.draw_geometries([pcl])
        open3d.io.write_point_cloud("pcl_{}.pcd".format("left" if is_left else "right"), pcl)

    def process_pcl(self):
        self.process.set_sgbm_parameters(self.num_disparity_slider.value() * 16,
                                         self.min_disparity_slider.value(),
                                         self.block_size_slider.value(),
                                         self.p1_slider.value(),
                                         self.p2_slider.value(),
                                         self.max_dif_slider.value(),
                                         self.uniqueness_slider.value(),
                                         self.speckle_slider.value()
                                         )
        mesh = self.process.process_image_batch(self.img1, self.img2, self.img3)
        open3d.visualization.draw_geometries([mesh])

    def show_image(self, image: np.ndarray):
        cv2.imshow("", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] %(levelname)s: %(message)s")
    app = QApplication(sys.argv)

    window = MainWindow()

    app.exec_()
