# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from find_object import *
from gui import Ui_MainWindow


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detector, matcher = init_feature('sift')
    # load object & calculate object feature
    image_path = "chip200.jpg"
    object_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    object_image = cv2.resize(object_image, (120, 120))
    object_feature, desc1 = detector.detectAndCompute(object_image, None)  # get feature of object
    show_corner = False # debug

    def run(self, camera_index="/media/hungtooc/Source/Public/applitcations/find-object/videos/demo-v0.4.mp4"):
        # capture from web cam
        cap = cv2.VideoCapture(camera_index)
        while True:
            ret, frame = cap.read()
            if ret:
                homography = find_homography(self.object_image, frame, self.detector, self.matcher, self.desc1, self.object_feature)
                if homography is not None:
                    object_coord, object_corners = get_object_coord(
                        self.object_image, frame, homography)
                    if object_coord:
                        cv2.circle(frame, object_coord, 4, (255, 0, 0), -1)
                        print(object_coord)
                        cv2.putText(frame, f"{object_coord}", object_coord,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if self.show_corner == True:
                    for corner in object_corners:
                        cv2.circle(frame, corner, 3, (0, 255, 0), -1)
                self.change_pixmap_signal.emit(frame)


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(w, h, Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(p)

def update_image(cv_img):
    """Updates the camera_label with a new opencv image"""
    qt_img = convert_cv_qt(cv_img)
    ui.camera_label.setPixmap(qt_img)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.thread = VideoThread()
    ui.thread.change_pixmap_signal.connect(update_image)
    ui.thread.start()
    
    MainWindow.show()
    sys.exit(app.exec_())