import cv2
import sys
import numpy as np
import math
import ui
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap

from Lib.prediction import predict_class
from Lib import train
from Models.XRN50.XRN50_Model import XRN50
from tensorflow.keras.applications import DenseNet169
from Models.XRN50 import config as cfg
from Models.DenseNet import config as Dense_cfg
from keras.layers import Input
from keras import Model

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changeTime = pyqtSignal(str)
    changeStage = pyqtSignal(str)
    changeStatus = pyqtSignal(str)
    def __init__(self, parent=None):
        QThread.__init__(self, parent)

        self.model = None
        self.source = None

    def run(self):
        self.ThreadActive = True
        # capture video
        if self.source:
            cap = cv2.VideoCapture(self.source)
            self.changeStatus.emit('Идет распознавание: Файл')
        else:
            cap = cv2.VideoCapture(0)
            self.changeStatus.emit('Идет распознавание: Камера')

        # set color for text capture
        text_color = (0, 255, 0)
        classes = ['Нагрев флюса', 'Плавление флюса', 'Плавление припоя', 'Стабилизация припоя']

        sec = 0
        time = str(sec) + ' секунд'
        # get some video params and set default text capture
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        text = "Waiting"
        pred = np.zeros((1,4))

        # frame processing > read image every second > predict its class >
        # > add capture (class, probability) > exit with 'q' or EOF
        #try:
        while cap.isOpened():
            frame_id = cap.get(1)
            ret, frame = cap.read()
            if not ret: 
                print("end of video")
                break

            if frame_id % math.floor(fps) == 0:
                sec = sec + 1
                time = str(sec) + ' секунд'
            if (frame_id % math.floor(fps) == 0) or frame_id == 1:
                pred = predict_class(self.model, frame, resize=(224, 224))
                text = "probabilities: " + str(pred)
                index = np.argmax(pred)
                stage = classes[index]
            frame = cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
            h, w, ch = frame.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_BGR888)
            p = convertToQtFormat.scaled(600, 600, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
            self.changeTime.emit(time)
            self.changeStage.emit(stage)
        #except:
        #    cap.release()
        cap.release()
        self.stop()

    def stop(self):
        self.ThreadActive = False
        self.changeStatus.emit('Процесс распознавания остановлен')
        self.quit()

    def timer(self):
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0/self.frame_rate)


class App(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.browse.clicked.connect(self.browsefiles)
        self.stopButton.clicked.connect(self.CancelFeed)
        self.startButton.clicked.connect(self.initThread)
        self.prepareNN()

    def prepareNN(self):
        #self.model = XRN50().build(input_shape = (cfg.IMAGE_W, cfg.IMAGE_H, 3), num_classes = cfg.NUM_CLASSES, summary = False)
        #train.prepare_model(self.model, "XRN50_best.h5")
        inputs = Input(shape=(Dense_cfg.IMAGE_W, Dense_cfg.IMAGE_H, 3))
        outputs = DenseNet169(include_top=True, weights=None, classes=Dense_cfg.NUM_CLASSES)(inputs)
        self.model = Model(inputs, outputs, name = 'DenseNet169')
        train.prepare_model(self.model, "DenseNet169_best.h5")
        self.valueImageSize.setText(str(Dense_cfg.IMAGE_W) + 'x' + str(Dense_cfg.IMAGE_H))
        self.valueModel.setText(self.model.name)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.imagePlaceholder.setPixmap(QPixmap.fromImage(image))

    def setTime(self,str):
        self.valueTime.setText(str)

    def setStage(self,str):
        self.valueStage.setText(str)

    def setThreadStatus(self, str):
        self.labelThread.setText(str)

    def initThread(self):
        # create a label
        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.changeTime.connect(self.setTime)
        self.th.changeStage.connect(self.setStage)
        self.th.changeStatus.connect(self.setThreadStatus)
        self.th.model = self.model
        if self.radioSourceFile.isChecked():
            self.th.source = self.filename.text()
            print(self.filename.text())
        self.th.start()
        self.show()

    def browsefiles(self):
        fname=QFileDialog.getOpenFileName(self, 'Открыть файл', 'Видео индукционной пайки', 'Video Files (*.mp4 *.flv *.ts *.mts *.avi)')
        self.filename.setText(fname[0])

    def CancelFeed(self):
        self.th.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())