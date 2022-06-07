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
        MainWindow.resize(1072, 684)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(220, 10, 491, 631))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.camera_label = QtWidgets.QLabel(self.frame)
        self.camera_label.setGeometry(QtCore.QRect(0, 0, 480, 640))
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
        self.groupBox_2.setGeometry(QtCore.QRect(720, 10, 341, 481))
        self.groupBox_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.n_objectA = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_objectA.setGeometry(QtCore.QRect(80, 30, 98, 31))
        self.n_objectA.setObjectName("n_objectA")
        self.n_objectB = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_objectB.setGeometry(QtCore.QRect(80, 60, 98, 31))
        self.n_objectB.setObjectName("n_objectB")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(10, 40, 64, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(10, 70, 64, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setScaledContents(False)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.n_objectC = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_objectC.setGeometry(QtCore.QRect(80, 90, 98, 31))
        self.n_objectC.setObjectName("n_objectC")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(10, 100, 64, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_7.setScaledContents(False)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(170, 40, 64, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_8.setFont(font)
        self.label_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_8.setTextFormat(QtCore.Qt.RichText)
        self.label_8.setScaledContents(False)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.n_total = QtWidgets.QLCDNumber(self.groupBox_2)
        self.n_total.setGeometry(QtCore.QRect(240, 30, 98, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.n_total.setFont(font)
        self.n_total.setSmallDecimalPoint(False)
        self.n_total.setDigitCount(3)
        self.n_total.setProperty("value", 0.0)
        self.n_total.setProperty("intValue", 0)
        self.n_total.setObjectName("n_total")
        self.line_4 = QtWidgets.QFrame(self.groupBox_2)
        self.line_4.setGeometry(QtCore.QRect(10, 130, 321, 16))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 211, 631))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_5.setGeometry(QtCore.QRect(0, 20, 201, 141))
        self.groupBox_5.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignHCenter)
        self.groupBox_5.setObjectName("groupBox_5")
        self.label_current_object = QtWidgets.QLabel(self.groupBox_5)
        self.label_current_object.setGeometry(QtCore.QRect(50, 30, 100, 100))
        self.label_current_object.setText("")
        self.label_current_object.setPixmap(QtGui.QPixmap("images/object_01.png"))
        self.label_current_object.setScaledContents(True)
        self.label_current_object.setObjectName("label_current_object")
        self.temp_label = QtWidgets.QLabel(self.tab_2)
        self.temp_label.setGeometry(QtCore.QRect(10, 180, 88, 17))
        self.temp_label.setObjectName("temp_label")
        self.label_current_object_type = QtWidgets.QLabel(self.tab_2)
        self.label_current_object_type.setGeometry(QtCore.QRect(100, 180, 88, 17))
        self.label_current_object_type.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_object_type.setObjectName("label_current_object_type")
        self.label_current_object_pos = QtWidgets.QLabel(self.tab_2)
        self.label_current_object_pos.setGeometry(QtCore.QRect(100, 210, 88, 17))
        self.label_current_object_pos.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_object_pos.setObjectName("label_current_object_pos")
        self.temp_label_2 = QtWidgets.QLabel(self.tab_2)
        self.temp_label_2.setGeometry(QtCore.QRect(10, 210, 88, 17))
        self.temp_label_2.setObjectName("temp_label_2")
        self.temp_label_3 = QtWidgets.QLabel(self.tab_2)
        self.temp_label_3.setGeometry(QtCore.QRect(10, 240, 88, 17))
        self.temp_label_3.setObjectName("temp_label_3")
        self.label_current_object_size = QtWidgets.QLabel(self.tab_2)
        self.label_current_object_size.setGeometry(QtCore.QRect(100, 240, 88, 17))
        self.label_current_object_size.setAlignment(QtCore.Qt.AlignCenter)
        self.label_current_object_size.setObjectName("label_current_object_size")
        self.checkBox_markcorner = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_markcorner.setGeometry(QtCore.QRect(10, 290, 109, 24))
        self.checkBox_markcorner.setChecked(True)
        self.checkBox_markcorner.setObjectName("checkBox_markcorner")
        self.checkBox_markcenter = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_markcenter.setGeometry(QtCore.QRect(10, 270, 109, 24))
        self.checkBox_markcenter.setChecked(True)
        self.checkBox_markcenter.setObjectName("checkBox_markcenter")
        self.line_2 = QtWidgets.QFrame(self.tab_2)
        self.line_2.setGeometry(QtCore.QRect(10, 260, 171, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.checkBox_markdebug = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_markdebug.setGeometry(QtCore.QRect(10, 310, 109, 24))
        self.checkBox_markdebug.setChecked(False)
        self.checkBox_markdebug.setObjectName("checkBox_markdebug")
        self.tabWidget.addTab(self.tab_2, "")
        self.info = QtWidgets.QWidget()
        self.info.setObjectName("info")
        self.groupBox_3 = QtWidgets.QGroupBox(self.info)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 0, 191, 431))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(70, 40, 100, 100))
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap("images/object_01.png"))
        self.label_9.setScaledContents(True)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(70, 150, 100, 100))
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap("images/object_02.png"))
        self.label_10.setScaledContents(True)
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(70, 260, 100, 100))
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap("images/object_03.png"))
        self.label_12.setScaledContents(True)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(0, 80, 67, 17))
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setGeometry(QtCore.QRect(0, 190, 67, 17))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setGeometry(QtCore.QRect(0, 300, 67, 17))
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.tabWidget.addTab(self.info, "")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(930, 590, 131, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(780, 590, 131, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1072, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "find object"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Collected"))
        self.label_3.setText(_translate("MainWindow", "Type A:"))
        self.label_5.setText(_translate("MainWindow", "Type B:"))
        self.label_7.setText(_translate("MainWindow", "Type C:"))
        self.label_8.setText(_translate("MainWindow", "Total:"))
        self.groupBox_5.setTitle(_translate("MainWindow", "current object"))
        self.temp_label.setText(_translate("MainWindow", "Object type:"))
        self.label_current_object_type.setText(_translate("MainWindow", "None"))
        self.label_current_object_pos.setText(_translate("MainWindow", "(x,y)"))
        self.temp_label_2.setText(_translate("MainWindow", "Position:"))
        self.temp_label_3.setText(_translate("MainWindow", "Size:"))
        self.label_current_object_size.setText(_translate("MainWindow", "(x,y)"))
        self.checkBox_markcorner.setText(_translate("MainWindow", "Mark corner"))
        self.checkBox_markcenter.setText(_translate("MainWindow", "Mark center"))
        self.checkBox_markdebug.setText(_translate("MainWindow", "debug"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Detection"))
        self.label_13.setText(_translate("MainWindow", "type A"))
        self.label_14.setText(_translate("MainWindow", "type B"))
        self.label_15.setText(_translate("MainWindow", "type C"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.info), _translate("MainWindow", "objects"))
        self.pushButton.setText(_translate("MainWindow", "Shutdown"))
        self.pushButton_4.setText(_translate("MainWindow", "Restart"))

