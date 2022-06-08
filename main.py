# -*- coding: utf-8 -*-

import time
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
    change_lcdtype_signal = pyqtSignal(int)
    detector, matcher = init_feature('sift')
    # load object & calculate object feature
    image_paths = ["images/object_01.png", "images/object_02.png", "images/object_03.png"]
    object_images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]
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
        if cap.isOpened():
            ret, frame = cap.read()
            
        global mark_center, mark_corner, debug, image_resolution, collected_flag,min_size, unknown_threshold, blur, erode, dilation, closing, adaptive, collect_line
        while True:
            ret, frame = cap.read()
            time.sleep(0.00001)
            if ret:
                circle_stat, closing_image = get_circle(frame,min_size,blur_kernel=(blur, blur),adaptive_size=adaptive, erode_kernel=(erode, erode), dilation_kernel=(dilation, dilation), closing_kernel=(closing, closing))
                
                if circle_stat is not None:
                    
                    circle_object = frame[circle_stat[1]:circle_stat[1] + circle_stat[3], circle_stat[0]:circle_stat[0] + circle_stat[2]].copy()
                    
                    
                    ## classify
                    output = getobject(circle_object, self.object_images, self.detector, self.matcher, self.object_features, unknown_threshold=unknown_threshold,debug=debug)
                    
                    if debug:
                        frame = cv.addWeighted(frame, 1.0, closing_image,0.5, 0.0)

                    if output is not None:
                        
                        bestest_object, confident_score, (object_coord, object_corners) = output[0]
                        object_type = output[1]
                        if object_type != -1:
                            center = (circle_stat[0] +  circle_stat[2]//2, circle_stat[1] +  circle_stat[3]//2)
                            self.change_currentobjectpos_signal.emit(str(center))
                            self.change_currentobjectsize_signal.emit(f"{circle_stat[2]}x{circle_stat[3]}")

                            if mark_center:          
                                cv2.circle(frame, center, 4, (255, 0, 0), -1)
                                cv2.putText(frame, f"{center}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)
                            if mark_corner:
                                cv2.rectangle(frame, (circle_stat[0], circle_stat[1]), (circle_stat[0] + circle_stat[2], circle_stat[1] + circle_stat[3]), (255,255,0))

                            self.change_currentobjecttype_signal.emit(object_names[object_type])
                            self.change_currentobject_signal.emit(circle_object)
                            
                            # Collect
                            if center[1] > int(frame.shape[0]*(collect_line/100)) and not collected_flag:
                                self.change_lcdtype_signal.emit(object_type) #
                                collected_flag = True
                            if center[1] < int(frame.shape[0]*(collect_line/100)) and collected_flag:
                                collected_flag = False
                frame = cv2.line(frame, (int(frame.shape[1]*(collect_line/100)), 0), (int(frame.shape[1]*(collect_line/100)), frame.shape[0]), (255,255,255))
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

def add_collected(type):
    if type == 0:
        ui.n_objectA.display(ui.n_objectA.value()+1)
    elif type == 1:
        ui.n_objectB.display(ui.n_objectB.value()+1)
    elif type == 2:
        ui.n_objectC.display(ui.n_objectC.value()+1)
    ui.n_total.display(ui.n_total.value()+1)

def set_minsize(value):
    global min_size
    min_size = value

def set_unknownthreshold(value):
    global unknown_threshold
    unknown_threshold = value

def set_blur(value):
    global blur
    blur = value

def set_adaptive(value):
    global adaptive
    adaptive = value

def set_erode(value):
    global erode
    erode = value

def set_dilation(value):
    global dilation
    dilation = value

def set_closing(value):
    global closing
    closing = value

def set_collect_line(value):
    global collect_line
    collect_line = value

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
    min_size = 100
    unknown_threshold = 20
    blur =5
    adaptive=2
    erode=2
    dilation =5
    closing=10
    collect_line = 50
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    
    # slider
    ui.horizontalSlider_minsize.setValue(min_size)
    ui.horizontalSlider_unknownthreshold.setValue(unknown_threshold)
    ui.horizontalSlider_blur.setValue(blur)
    ui.horizontalSlider_adaptive.setValue(adaptive)
    ui.horizontalSlider_erode.setValue(erode)
    ui.horizontalSlider_dilation.setValue(dilation)
    ui.horizontalSlider_closing.setValue(closing)
    ui.horizontalSlider_collectline.setValue(collect_line)
    ui.horizontalSlider_minsize.valueChanged.connect(set_minsize)
    ui.horizontalSlider_unknownthreshold.valueChanged.connect(set_unknownthreshold)
    ui.horizontalSlider_blur.valueChanged.connect(set_blur)
    ui.horizontalSlider_adaptive.valueChanged.connect(set_adaptive)
    ui.horizontalSlider_erode.valueChanged.connect(set_erode)
    ui.horizontalSlider_dilation.valueChanged.connect(set_dilation)
    ui.horizontalSlider_closing.valueChanged.connect(set_closing)
    ui.horizontalSlider_collectline.valueChanged.connect(set_collect_line)
    
    mark_center = ui.checkBox_markcenter.isChecked()
    mark_corner = ui.checkBox_markcorner.isChecked()
    debug = ui.checkBox_markdebug.isChecked()
    collected_flag = False
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

    #lcd
    ui.thread.change_lcdtype_signal.connect(add_collected)
    ui.thread.start()

    MainWindow.show()
    sys.exit(app.exec_())
