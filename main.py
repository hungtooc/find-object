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
    change_debug_signal = pyqtSignal(np.ndarray)

    # text
    change_currentobjecttype_signal = pyqtSignal(str)
    change_currentobjectpos_signal = pyqtSignal(str)
    change_currentobjectsize_signal = pyqtSignal(str)

    detector, matcher = init_feature('sift')
    # load object & calculate object feature
    image_paths = ["images/object_01.png", "images/object_02.png", "images/object_03.png"]
    object_images = [cv2.imread(image_path) for image_path in image_paths]
    object_features = []
    for index, object_image in enumerate(object_images):
        feature = detector.detectAndCompute(object_image, None)
        kpts2, descs2 = feature
        img = cv.drawKeypoints(object_image,kpts2,object_image,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imwrite(f'temps/sift_keypoints_{index}.jpg',img)
        object_features.append(feature)
    # object_features = [detector.detectAndCompute(object_image, None) for object_image in object_images]
    

    def run(self, camera_url="videos/demo-v0.2.mp4"):
        # camera_url="videos/demo-v0.2.mp4"
        # camera_url=0
        # camera_url=1
        cap = cv2.VideoCapture(camera_url)
        global mark_center, mark_corner, debug, image_resolution
        while True:
            ret, frame = cap.read()
            if ret:
                circle_stat = get_circle(frame)
                if circle_stat is not None:
                    center = (circle_stat[0] +  circle_stat[2]//2, circle_stat[1] +  circle_stat[3]//2)
                    self.change_currentobjectpos_signal.emit(str(center))
                    self.change_currentobjectsize_signal.emit(f"{circle_stat[2]}x{circle_stat[3]}")
                    
                    circle_object = frame[circle_stat[1]:circle_stat[1] + circle_stat[3], circle_stat[0]:circle_stat[0] + circle_stat[2]].copy()
                    if mark_center:          
                        cv2.circle(frame, center, 4, (255, 0, 0), -1)
                        cv2.putText(frame, f"{center}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
                    if mark_corner:
                        cv2.rectangle(frame, (circle_stat[0], circle_stat[1]), (circle_stat[0] + circle_stat[2], circle_stat[1] + circle_stat[3]), (255,255,0))
                    
                    ## classify
                    output = getobject(circle_object, self.object_images, self.detector, self.matcher, self.object_features, debug )
                    # print("type(output)", type(output))
                    if output is not None:
                        
                        bestest_object, confident_score, (object_coord, object_corners) = output[0]
                        object_type = output[1]
                        self.change_currentobjecttype_signal.emit(object_names[object_type])
                        self.change_currentobject_signal.emit(circle_object)
                        
                self.change_camera_signal.emit(frame)
            else: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


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

    
        

def update_currentobject(cv_img):
    qt_img = convert_cv_qt(cv_img)
    ui.label_current_object.setPixmap(qt_img)
    

def update_currentobjecttype(object_name):
    ui.label_current_object_type.setText(object_name)

def update_currentobjectpos(pos):
    ui.label_current_object_pos.setText(pos)

def update_currentobjectsize(pos):
    ui.label_current_object_size.setText(pos)

def set_debug(state):
    global debug
    debug = state

def set_mark_center(state):
    global mark_center
    mark_center = state

def set_mark_corner(state):
    global mark_corner
    mark_corner = state

if __name__ == "__main__":
    object_names = ["A", "B", "C"]
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    mark_center = ui.checkBox_markcenter.isChecked()
    mark_corner = ui.checkBox_markcorner.isChecked()
    debug = ui.checkBox_markdebug.isChecked()
    
    ui.thread = VideoThread()
    
    # update image
    ui.thread.change_camera_signal.connect(update_camera)
    ui.thread.change_currentobject_signal.connect(update_currentobject)
    # update text
    ui.checkBox_markdebug.toggled.connect(set_debug)
    ui.checkBox_markcenter.toggled.connect(set_mark_center)
    ui.checkBox_markcorner.toggled.connect(set_mark_corner)

    ui.thread.change_currentobjecttype_signal.connect(update_currentobjecttype)
    ui.thread.change_currentobjectpos_signal.connect(update_currentobjectpos)
    ui.thread.change_currentobjectsize_signal.connect(update_currentobjectsize)
    ui.thread.start()

    MainWindow.show()
    sys.exit(app.exec_())
