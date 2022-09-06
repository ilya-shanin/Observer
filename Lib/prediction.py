###############################################################
# predict single image class, predict frames class from video #
# 16.04.2022                                                  #
# v1.1 (17.04.2022)                                           #
###############################################################

import cv2
import os
import numpy as np
import math
from Lib.train import prepare_model
from Models.XRN50 import config as cfg
from Models.UNet import config as cfgU
from tensorflow.keras.applications import DenseNet169
from Models.DenseNet import config as Dense_cfg
from keras.layers import Input
from keras import Model


def predict_class_from_file(model, image_path, resize = (cfg.IMAGE_W, cfg.IMAGE_H)):
    # read image > resize it > predict > return results    
    image = cv2.imread(image_path)
    image = cv2.resize(image, resize)
    image = np.array(image)/255.
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    return preds

def predict_class(model, frame, resize = (Dense_cfg.IMAGE_W, Dense_cfg.IMAGE_H)):
    # read image > resize it > predict > return results    
    image = cv2.resize(frame, resize)
    image = np.array(image)/255.
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    np.around(preds, 2, preds)
    return preds

def mask_image(model, frame, resize = (cfgU.IMAGE_W, cfgU.IMAGE_H)):
    image = cv2.resize(frame, resize)
    image = np.array(image)/255.
    mask = model.get_mask(image)
    return mask

def process_video_masked(model_classifier, file_name, model_segmentator=None):
    # set color for text capture
    text_color = (0, 255, 0)

    # capture video
    cap = cv2.VideoCapture(file_name)

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
        if not ret: break

        if (frame_id % math.floor(fps) == 0) or frame_id == 1:
            #mask = mask_image(model_segmentator ,frame)
            pred = predict_class(model_classifier, frame)
            text = "probabilities: " + str(pred)
        #frame = np.concatenate((frame, cv2.resize(mask, (width, height))), axis = 1)
        frame = cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, text_color, 2)
        cv2.imshow(file_name, frame)
        #cv2.imshow("masked", mask)
        if cv2.waitKey(33) & 0xFF == ord('q'): break
    #except:
    #    cap.release()
    cap.release()
    cv2.destroyAllWindows()

#process_video(XResNet, UNetModel, 'Videos/Flux_Heat/Flux_Heat.mp4')
#inputs = Input(shape=(Dense_cfg.IMAGE_W, Dense_cfg.IMAGE_H, 3))
#outputs = DenseNet169(include_top=True, weights=None, classes=Dense_cfg.NUM_CLASSES)(inputs)
#model = Model(inputs, outputs, name = 'DenseNet169')
#prepare_model(model, "DenseNet169_best.h5")
#process_video_masked(model, 'Videos/Flux_Heat/Flux_Heat.mp4')