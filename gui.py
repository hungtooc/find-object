# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'windows.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1249, 567)
        MainWindow.setFocusPolicy(QtCore.Qt.WheelFocus)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/object_01.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(290, 10, 661, 511))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.camera_label = QtWidgets.QLabel(self.frame)
        self.camera_label.setGeometry(QtCore.QRect(10, 10, 640, 491))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera_label.sizePolicy().hasHeightForWidth())
        self.camera_label.setSizePolicy(sizePolicy)
        self.camera_label.setText("")
        self.camera_label.setPixmap(QtGui.QPixmap("../../../../../../home/hungtooc/Pictures/7e5af3311146d8188157.jpg"))
        self.camera_label.setScaledContents(True)
        self.camera_label.setObjectName("camera_label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(960, 0, 281, 401))
        self.groupBox_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.n_objectA = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_objectA.setGeometry(QtCore.QRect(90, 40, 100, 31))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.n_objectA.setFont(font)
        self.n_objectA.setFrameShape(QtWidgets.QFrame.Box)
        self.n_objectA.setSmallDecimalPoint(False)
        self.n_objectA.setDigitCount(6)
        self.n_objectA.setObjectName("n_objectA")
        self.n_objectB = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_objectB.setGeometry(QtCore.QRect(90, 80, 100, 31))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.n_objectB.setFont(font)
        self.n_objectB.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.n_objectB.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.n_objectB.setFrameShape(QtWidgets.QFrame.Box)
        self.n_objectB.setSmallDecimalPoint(False)
        self.n_objectB.setDigitCount(6)
        self.n_objectB.setObjectName("n_objectB")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(10, 50, 66, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(10, 90, 66, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setScaledContents(False)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.n_objectC = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_objectC.setGeometry(QtCore.QRect(90, 120, 100, 31))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.n_objectC.setFont(font)
        self.n_objectC.setFrameShape(QtWidgets.QFrame.Box)
        self.n_objectC.setSmallDecimalPoint(False)
        self.n_objectC.setDigitCount(6)
        self.n_objectC.setObjectName("n_objectC")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(10, 130, 66, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_7.setScaledContents(False)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(10, 170, 66, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_8.setTextFormat(QtCore.Qt.RichText)
        self.label_8.setScaledContents(False)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.n_total = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_total.setGeometry(QtCore.QRect(90, 160, 100, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.n_total.setFont(font)
        self.n_total.setSmallDecimalPoint(False)
        self.n_total.setDigitCount(6)
        self.n_total.setProperty("value", 0.0)
        self.n_total.setProperty("intValue", 0)
        self.n_total.setObjectName("n_total")
        self.line_4 = QtWidgets.QFrame(self.groupBox_2)
        self.line_4.setGeometry(QtCore.QRect(10, 230, 290, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton.setGeometry(QtCore.QRect(20, 240, 141, 31))
        self.radioButton.setObjectName("radioButton")
        self.pushButton_stopbelt = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_stopbelt.setGeometry(QtCore.QRect(30, 280, 101, 31))
        self.pushButton_stopbelt.setObjectName("pushButton_stopbelt")
        self.pushButton_collect = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_collect.setGeometry(QtCore.QRect(160, 280, 91, 31))
        self.pushButton_collect.setObjectName("pushButton_collect")
        self.horizontalSlider_collectline = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider_collectline.setGeometry(QtCore.QRect(10, 360, 260, 16))
        self.horizontalSlider_collectline.setMinimum(0)
        self.horizontalSlider_collectline.setProperty("value", 50)
        self.horizontalSlider_collectline.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_collectline.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_collectline.setTickInterval(10)
        self.horizontalSlider_collectline.setObjectName("horizontalSlider_collectline")
        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        self.label_18.setGeometry(QtCore.QRect(10, 340, 112, 17))
        self.label_18.setObjectName("label_18")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 271, 511))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 20, 231, 141))
        self.groupBox_5.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.groupBox_5.setObjectName("groupBox_5")
        self.label_current_object = QtWidgets.QLabel(self.groupBox_5)
        self.label_current_object.setGeometry(QtCore.QRect(70, 30, 100, 100))
        self.label_current_object.setText("")
        self.label_current_object.setPixmap(QtGui.QPixmap("images/object_01.png"))
        self.label_current_object.setScaledContents(True)
        self.label_current_object.setObjectName("label_current_object")
        self.temp_label = QtWidgets.QLabel(self.tab_2)
        self.temp_label.setGeometry(QtCore.QRect(10, 180, 88, 17))
        self.temp_label.setObjectName("temp_label")
        self.label_current_object_type = QtWidgets.QLabel(self.tab_2)
        self.label_current_object_type.setGeometry(QtCore.QRect(170, 180, 88, 17))
        self.label_current_object_type.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_object_type.setObjectName("label_current_object_type")
        self.label_current_object_pos = QtWidgets.QLabel(self.tab_2)
        self.label_current_object_pos.setGeometry(QtCore.QRect(170, 210, 88, 17))
        self.label_current_object_pos.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_object_pos.setObjectName("label_current_object_pos")
        self.temp_label_2 = QtWidgets.QLabel(self.tab_2)
        self.temp_label_2.setGeometry(QtCore.QRect(10, 210, 88, 17))
        self.temp_label_2.setObjectName("temp_label_2")
        self.temp_label_3 = QtWidgets.QLabel(self.tab_2)
        self.temp_label_3.setGeometry(QtCore.QRect(10, 240, 88, 17))
        self.temp_label_3.setObjectName("temp_label_3")
        self.label_current_object_size = QtWidgets.QLabel(self.tab_2)
        self.label_current_object_size.setGeometry(QtCore.QRect(170, 240, 88, 17))
        self.label_current_object_size.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_object_size.setObjectName("label_current_object_size")
        self.line_2 = QtWidgets.QFrame(self.tab_2)
        self.line_2.setGeometry(QtCore.QRect(10, 260, 251, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.info = QtWidgets.QWidget()
        self.info.setObjectName("info")
        self.groupBox_3 = QtWidgets.QGroupBox(self.info)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 0, 241, 361))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(110, 30, 100, 100))
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap("images/object_01.png"))
        self.label_9.setScaledContents(True)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(110, 140, 100, 100))
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap("images/object_02.png"))
        self.label_10.setScaledContents(True)
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(110, 250, 100, 100))
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap("images/object_03.png"))
        self.label_12.setScaledContents(True)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(20, 70, 67, 17))
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setGeometry(QtCore.QRect(20, 180, 67, 17))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setGeometry(QtCore.QRect(20, 290, 67, 17))
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.line = QtWidgets.QFrame(self.info)
        self.line.setGeometry(QtCore.QRect(20, 370, 221, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.tabWidget.addTab(self.info, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_4.setGeometry(QtCore.QRect(0, 10, 270, 251))
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        self.label_4.setGeometry(QtCore.QRect(10, 40, 59, 17))
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.groupBox_4)
        self.label_6.setGeometry(QtCore.QRect(10, 120, 59, 17))
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.label_11 = QtWidgets.QLabel(self.groupBox_4)
        self.label_11.setGeometry(QtCore.QRect(10, 160, 59, 17))
        self.label_11.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.label_16 = QtWidgets.QLabel(self.groupBox_4)
        self.label_16.setGeometry(QtCore.QRect(10, 200, 59, 17))
        self.label_16.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_16.setObjectName("label_16")
        self.horizontalSlider_blur = QtWidgets.QSlider(self.groupBox_4)
        self.horizontalSlider_blur.setGeometry(QtCore.QRect(10, 60, 250, 16))
        self.horizontalSlider_blur.setMinimum(3)
        self.horizontalSlider_blur.setMaximum(10)
        self.horizontalSlider_blur.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_blur.setObjectName("horizontalSlider_blur")
        self.horizontalSlider_erode = QtWidgets.QSlider(self.groupBox_4)
        self.horizontalSlider_erode.setGeometry(QtCore.QRect(10, 140, 250, 16))
        self.horizontalSlider_erode.setMinimum(2)
        self.horizontalSlider_erode.setMaximum(10)
        self.horizontalSlider_erode.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_erode.setObjectName("horizontalSlider_erode")
        self.horizontalSlider_dilation = QtWidgets.QSlider(self.groupBox_4)
        self.horizontalSlider_dilation.setGeometry(QtCore.QRect(10, 180, 250, 16))
        self.horizontalSlider_dilation.setMinimum(3)
        self.horizontalSlider_dilation.setMaximum(10)
        self.horizontalSlider_dilation.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_dilation.setObjectName("horizontalSlider_dilation")
        self.horizontalSlider_closing = QtWidgets.QSlider(self.groupBox_4)
        self.horizontalSlider_closing.setGeometry(QtCore.QRect(10, 220, 250, 16))
        self.horizontalSlider_closing.setMinimum(3)
        self.horizontalSlider_closing.setMaximum(15)
        self.horizontalSlider_closing.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_closing.setObjectName("horizontalSlider_closing")
        self.horizontalSlider_adaptive = QtWidgets.QSlider(self.groupBox_4)
        self.horizontalSlider_adaptive.setGeometry(QtCore.QRect(10, 100, 250, 16))
        self.horizontalSlider_adaptive.setMinimum(1)
        self.horizontalSlider_adaptive.setMaximum(5)
        self.horizontalSlider_adaptive.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_adaptive.setObjectName("horizontalSlider_adaptive")
        self.label_17 = QtWidgets.QLabel(self.groupBox_4)
        self.label_17.setGeometry(QtCore.QRect(10, 80, 67, 17))
        self.label_17.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_17.setObjectName("label_17")
        self.checkBox_markdebug = QtWidgets.QCheckBox(self.tab)
        self.checkBox_markdebug.setGeometry(QtCore.QRect(10, 400, 140, 24))
        self.checkBox_markdebug.setChecked(False)
        self.checkBox_markdebug.setObjectName("checkBox_markdebug")
        self.checkBox_markcorner = QtWidgets.QCheckBox(self.tab)
        self.checkBox_markcorner.setGeometry(QtCore.QRect(10, 440, 140, 24))
        self.checkBox_markcorner.setChecked(True)
        self.checkBox_markcorner.setObjectName("checkBox_markcorner")
        self.checkBox_markcenter = QtWidgets.QCheckBox(self.tab)
        self.checkBox_markcenter.setGeometry(QtCore.QRect(10, 420, 140, 24))
        self.checkBox_markcenter.setChecked(True)
        self.checkBox_markcenter.setObjectName("checkBox_markcenter")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(0, 280, 270, 111))
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(60, 30, 71, 21))
        self.label.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(30, 70, 101, 21))
        self.label_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.label_19 = QtWidgets.QLabel(self.groupBox)
        self.label_19.setGeometry(QtCore.QRect(185, 35, 12, 17))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        self.label_20.setGeometry(QtCore.QRect(186, 75, 12, 17))
        self.label_20.setObjectName("label_20")
        self.spinBox_topleft_x = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_topleft_x.setGeometry(QtCore.QRect(130, 30, 48, 26))
        self.spinBox_topleft_x.setMinimum(-200)
        self.spinBox_topleft_x.setMaximum(200)
        self.spinBox_topleft_x.setProperty("value", 0)
        self.spinBox_topleft_x.setObjectName("spinBox_topleft_x")
        self.spinBox_topleft_y = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_topleft_y.setGeometry(QtCore.QRect(200, 30, 48, 26))
        self.spinBox_topleft_y.setMinimum(-200)
        self.spinBox_topleft_y.setMaximum(200)
        self.spinBox_topleft_y.setProperty("value", 0)
        self.spinBox_topleft_y.setObjectName("spinBox_topleft_y")
        self.spinBox_bottomright_x = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_bottomright_x.setGeometry(QtCore.QRect(130, 70, 48, 26))
        self.spinBox_bottomright_x.setMinimum(-200)
        self.spinBox_bottomright_x.setMaximum(200)
        self.spinBox_bottomright_x.setProperty("value", 0)
        self.spinBox_bottomright_x.setObjectName("spinBox_bottomright_x")
        self.spinBox_bottomright_y = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_bottomright_y.setGeometry(QtCore.QRect(200, 70, 48, 26))
        self.spinBox_bottomright_y.setMinimum(-200)
        self.spinBox_bottomright_y.setMaximum(200)
        self.spinBox_bottomright_y.setProperty("value", 0)
        self.spinBox_bottomright_y.setObjectName("spinBox_bottomright_y")
        self.tabWidget.addTab(self.tab, "")
        self.pushButton_shutdown = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_shutdown.setGeometry(QtCore.QRect(1110, 480, 131, 41))
        icon = QtGui.QIcon.fromTheme("shut")
        self.pushButton_shutdown.setIcon(icon)
        self.pushButton_shutdown.setObjectName("pushButton_shutdown")
        self.pushButton_restart = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_restart.setGeometry(QtCore.QRect(960, 480, 131, 41))
        self.pushButton_restart.setObjectName("pushButton_restart")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1249, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "find object"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Collected"))
        self.label_3.setText(_translate("MainWindow", "Type A:"))
        self.label_5.setText(_translate("MainWindow", "Type B:"))
        self.label_7.setText(_translate("MainWindow", "Type C:"))
        self.label_8.setText(_translate("MainWindow", "Total:"))
        self.radioButton.setText(_translate("MainWindow", "Automatic "))
        self.pushButton_stopbelt.setText(_translate("MainWindow", "Stop Belt"))
        self.pushButton_collect.setText(_translate("MainWindow", "Collect"))
        self.label_18.setText(_translate("MainWindow", "Collect position"))
        self.groupBox_5.setTitle(_translate("MainWindow", "current object"))
        self.temp_label.setText(_translate("MainWindow", "Object type:"))
        self.label_current_object_type.setText(_translate("MainWindow", "None"))
        self.label_current_object_pos.setText(_translate("MainWindow", "(x,y)"))
        self.temp_label_2.setText(_translate("MainWindow", "Position:"))
        self.temp_label_3.setText(_translate("MainWindow", "Size:"))
        self.label_current_object_size.setText(_translate("MainWindow", "(x,y)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Detection"))
        self.label_13.setText(_translate("MainWindow", "Type A"))
        self.label_14.setText(_translate("MainWindow", "Type B"))
        self.label_15.setText(_translate("MainWindow", "type C"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.info), _translate("MainWindow", "Infomation"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Filter configs"))
        self.label_4.setText(_translate("MainWindow", "Blur"))
        self.label_6.setText(_translate("MainWindow", "Erode"))
        self.label_11.setText(_translate("MainWindow", "Dilation"))
        self.label_16.setText(_translate("MainWindow", "Closing"))
        self.label_17.setText(_translate("MainWindow", "Adaptive"))
        self.checkBox_markdebug.setText(_translate("MainWindow", "Show debugs"))
        self.checkBox_markcorner.setText(_translate("MainWindow", "Mark corner"))
        self.checkBox_markcenter.setText(_translate("MainWindow", "Mark center"))
        self.groupBox.setTitle(_translate("MainWindow", "Coordinates"))
        self.label.setText(_translate("MainWindow", "Top left:"))
        self.label_2.setText(_translate("MainWindow", "Bottom right:"))
        self.label_19.setText(_translate("MainWindow", "X"))
        self.label_20.setText(_translate("MainWindow", "X"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Configs"))
        self.pushButton_shutdown.setText(_translate("MainWindow", "Shutdown"))
        self.pushButton_restart.setText(_translate("MainWindow", "Restart"))

