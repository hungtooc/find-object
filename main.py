# -*- coding: utf-8 -*-


import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from lib import *
from gui import Ui_MainWindow
import sys

class VideoThread(QThread):
    #image
    change_camera_signal = pyqtSignal(np.ndarray)
    change_currentobject_signal = pyqtSignal(np.ndarray)

    # text
    change_currentobjecttype_signal = pyqtSignal(str)
    change_currentobjectpos_signal = pyqtSignal(str)


    detector, matcher = init_feature('sift')
    # load object & calculate object feature
    image_path = "images/object_demo.jpg"
    object_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    object_image = cv2.resize(object_image, (120, 120))
    object_feature, desc1 = detector.detectAndCompute(object_image, None)  # get feature of object
    show_corner = False  # debug
    image_paths = ["images/object_01.png", "images/object_02.png", "images/object_demo.jpg"]
    object_images = [cv2.imread(image_path) for image_path in image_paths]
    object_features = []
    for object_image in object_images:
        object_features.append(detector.detectAndCompute(object_image, None))
    # object_features = [detector.detectAndCompute(object_image, None) for object_image in object_images]
    

    def run(self, camera_index=2):
        # capture from web cam
        cap = cv2.VideoCapture(camera_index)
        global mark_center, mark_corner
        while True:
            ret, frame = cap.read()
            if ret:
                # detected_object, object_type, _ = getobject(frame, self.object_images, self.detector, self.matcher, self.object_features)
                output = getobject(frame, self.object_images, self.detector, self.matcher, self.object_features)
                # print("type(output)", type(output))
                if output is not None:
                    # print("output", output)
                    bestest_object, confident_score, (object_coord, object_corners) = output[0]
                    object_type = output[1]
                    print("detected_object", object_type)
                    if bestest_object is not None:
                        # cv2.imwrite("detected_object.jpg", bestest_object)
                        bestest_object = cv2.resize(bestest_object, (240,240))
                        self.change_currentobject_signal.emit(bestest_object)
                        self.change_currentobjecttype_signal.emit(object_names[object_type])
                        
                    # homography = find_homography(
                    #     self.object_image, frame, self.detector, self.matcher, self.desc1, self.object_feature)
                    # if homography is not None:
                        # object_coord, object_corners = get_object_coord(self.object_image, frame, homography)
                    if object_coord:
                        self.change_currentobjectpos_signal.emit(str(object_coord))
                        if mark_center:
                            print("check box center True")
                            cv2.circle(frame, object_coord, 4, (255, 0, 0), -1)
                            # cv2.putText(frame, f"{object_coord}", object_coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                        else:
                            print("check box center False")
                        if mark_corner:
                            print("check box corner True")
                            for corner in object_corners:
                                cv2.circle(frame, corner, 3, (0, 255, 0), -1)
                        else:
                            print("check box corner False")
                self.change_camera_signal.emit(frame)


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(
        rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(w, h, Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(p)


def update_camera(cv_img):
    """Updates the camera_label with a new opencv image"""
    global mark_center, mark_corner
    qt_img = convert_cv_qt(cv_img)
    ui.camera_label.setPixmap(qt_img)
    if ui.checkBox_markcenter.isChecked():
        
        mark_center = True
    else:
        mark_center = False
        
    if ui.checkBox_markcorner.isChecked():
        mark_corner = True
        
    else:
        mark_corner = False
        

def update_currentobject(cv_img):
    qt_img = convert_cv_qt(cv_img)
    ui.label_current_object.setPixmap(qt_img)
    

def update_currentobjecttype(object_name):
    ui.label_current_object_type.setText(object_name)

def update_currentobjectpos(pos):
    ui.label_current_object_pos.setText(pos)



if __name__ == "__main__":
    object_names = ["object a", "object b", "object c"]
    mark_center = True
    mark_corner = True
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.thread = VideoThread()
    
    # update image
    ui.thread.change_camera_signal.connect(update_camera)
    ui.thread.change_currentobject_signal.connect(update_currentobject)
    # update text
    ui.thread.change_currentobjecttype_signal.connect(update_currentobjecttype)
    ui.thread.change_currentobjectpos_signal.connect(update_currentobjectpos)
    ui.thread.start()

    MainWindow.show()
    sys.exit(app.exec_())
