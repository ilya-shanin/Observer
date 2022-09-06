# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Observer.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(703, 471)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widgetMain = QtWidgets.QWidget(self.centralwidget)
        self.widgetMain.setGeometry(QtCore.QRect(0, 0, 701, 461))
        self.widgetMain.setStyleSheet("background-color:rgb(238, 238, 238);")
        self.widgetMain.setObjectName("widgetMain")
        self.radioSourceCamera = QtWidgets.QRadioButton(self.widgetMain)
        self.radioSourceCamera.setGeometry(QtCore.QRect(30, 370, 82, 17))
        self.radioSourceCamera.setObjectName("radioSourceCamera")
        self.radioSourceFile = QtWidgets.QRadioButton(self.widgetMain)
        self.radioSourceFile.setEnabled(True)
        self.radioSourceFile.setGeometry(QtCore.QRect(120, 370, 82, 17))
        self.radioSourceFile.setChecked(True)
        self.radioSourceFile.setObjectName("radioSourceFile")
        self.labelTime = QtWidgets.QLabel(self.widgetMain)
        self.labelTime.setGeometry(QtCore.QRect(371, 274, 47, 13))
        self.labelTime.setStyleSheet("font-weight: bold;")
        self.labelTime.setObjectName("labelTime")
        self.labelImage = QtWidgets.QLabel(self.widgetMain)
        self.labelImage.setGeometry(QtCore.QRect(371, 229, 131, 16))
        self.labelImage.setStyleSheet("font-weight: bold;")
        self.labelImage.setObjectName("labelImage")
        self.labelModel = QtWidgets.QLabel(self.widgetMain)
        self.labelModel.setGeometry(QtCore.QRect(371, 208, 51, 16))
        self.labelModel.setStyleSheet("font-weight: bold;")
        self.labelModel.setObjectName("labelModel")
        self.labelStage = QtWidgets.QLabel(self.widgetMain)
        self.labelStage.setGeometry(QtCore.QRect(371, 250, 91, 16))
        self.labelStage.setStyleSheet("font-weight: bold;")
        self.labelStage.setScaledContents(False)
        self.labelStage.setObjectName("labelStage")
        self.imagePlaceholder = QtWidgets.QLabel(self.widgetMain)
        self.imagePlaceholder.setGeometry(QtCore.QRect(30, 30, 311, 261))
        self.imagePlaceholder.setStyleSheet("background-color: #f6f5f6;\n"
"border-radius: 10px;")
        self.imagePlaceholder.setText("")
        self.imagePlaceholder.setPixmap(QtGui.QPixmap("Data/Train/Flux_Heat/frame-0-00-33.57.jpg"))
        self.imagePlaceholder.setScaledContents(True)
        self.imagePlaceholder.setObjectName("imagePlaceholder")
        self.stopButton = QtWidgets.QPushButton(self.widgetMain)
        self.stopButton.setGeometry(QtCore.QRect(140, 310, 101, 41))
        self.stopButton.setStyleSheet("background-color: #c53653;\n"
"color: #f6f5f6;\n"
"border-radius: 10px;\n"
"font-size: 14px;\n"
"font-weight: 500;")
        self.stopButton.setObjectName("stopButton")
        self.valueModel = QtWidgets.QLabel(self.widgetMain)
        self.valueModel.setGeometry(QtCore.QRect(431, 210, 251, 16))
        self.valueModel.setText("")
        self.valueModel.setObjectName("valueModel")
        self.valueImageSize = QtWidgets.QLabel(self.widgetMain)
        self.valueImageSize.setGeometry(QtCore.QRect(510, 231, 171, 16))
        self.valueImageSize.setText("")
        self.valueImageSize.setObjectName("valueImageSize")
        self.valueStage = QtWidgets.QLabel(self.widgetMain)
        self.valueStage.setGeometry(QtCore.QRect(467, 251, 211, 16))
        self.valueStage.setText("")
        self.valueStage.setObjectName("valueStage")
        self.valueTime = QtWidgets.QLabel(self.widgetMain)
        self.valueTime.setGeometry(QtCore.QRect(430, 273, 251, 16))
        self.valueTime.setText("")
        self.valueTime.setObjectName("valueTime")
        self.filename = QtWidgets.QLineEdit(self.widgetMain)
        self.filename.setGeometry(QtCore.QRect(30, 400, 511, 31))
        self.filename.setObjectName("filename")
        self.browse = QtWidgets.QPushButton(self.widgetMain)
        self.browse.setGeometry(QtCore.QRect(540, 400, 75, 31))
        self.browse.setStyleSheet("border-radius: 0 10px 10px 0;\n"
"background-color: #4591d7;\n"
"color: #f6f5f6;\n"
"font-size: 14px;\n"
"font-weight: 400;")
        self.browse.setObjectName("browse")
        self.startButton = QtWidgets.QPushButton(self.widgetMain)
        self.startButton.setGeometry(QtCore.QRect(30, 310, 101, 41))
        self.startButton.setStyleSheet("background-color: #4591d7;\n"
"color: #f6f5f6;\n"
"border-radius: 10px;\n"
"font-size: 14px;\n"
"font-weight: 500;")
        self.startButton.setObjectName("startButton")
        self.labelThread = QtWidgets.QLabel(self.widgetMain)
        self.labelThread.setGeometry(QtCore.QRect(370, 40, 321, 141))
        self.labelThread.setStyleSheet("font-size: 14pt;")
        self.labelThread.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.labelThread.setObjectName("labelThread")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Наблюдение этапов"))
        self.radioSourceCamera.setText(_translate("MainWindow", "С камеры"))
        self.radioSourceFile.setText(_translate("MainWindow", "Из файла"))
        self.labelTime.setText(_translate("MainWindow", "Прошло:"))
        self.labelImage.setText(_translate("MainWindow", "Размер изображения:"))
        self.labelModel.setText(_translate("MainWindow", "Модель:"))
        self.labelStage.setText(_translate("MainWindow", "Текущий этап:"))
        self.stopButton.setText(_translate("MainWindow", "Остановить"))
        self.filename.setText(_translate("MainWindow", "C:\\Users\\Pseud\\Documents\\Python Scripts\\Observer\\Data\\Test\\2.mp4"))
        self.browse.setText(_translate("MainWindow", "Обзор"))
        self.startButton.setText(_translate("MainWindow", "Запустить"))
        self.labelThread.setText(_translate("MainWindow", "Распознавание не запущено"))