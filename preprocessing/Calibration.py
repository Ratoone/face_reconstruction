import glob

import numpy as np
import cv2


class Calibration:
    """
    Responsible for intrinsic and extrinsic camera calibration
    """

    def __init__(self):
        self.inner_x_grid = 6
        self.inner_y_grid = 9
        pass

    def intrinsic_calibration(self, image: np.ndarray):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.inner_x_grid * self.inner_y_grid, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.inner_x_grid, 0:self.inner_y_grid].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('../images/Calibratie 1/calibrationMiddle/*.jpg')

        for fname in images:
            image = cv2.imread(fname)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.inner_x_grid, self.inner_y_grid), None)
            print(ret)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(image, (self.inner_x_grid, self.inner_y_grid), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey()

        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    c = Calibration()
    image = cv2.imread("../images/Calibratie 1/calibrationMiddle/Calibratie 1_M_1.jpg")
    c.intrinsic_calibration(image)