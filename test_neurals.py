###############################################################
# testing networks                                            #
# 16.04.2022                                                  #
# v1.1 (17.04.2022)                                           #
###############################################################

import cv2
import os
import numpy as np
import math
from pathlib import Path

from Models.UNet.uNetModel import UNet
from Models.XRN50.XRN50_Model import XRN50
from Lib import train, callbacks
from Lib.utils import plot_hist

from Models.XRN50 import config as cfg
from Models.UNet import config as cfgU
from Models.EfficientNetB0 import config as EN_cfg
from Models.EfficientNetB4 import config as EN4_cfg
from Models.MobileNetV3Large import config as M3L_cfg
from Models.DenseNet import config as Dense_cfg
from Models.VGG19 import config as VGG19_cfg
from Models.Xception import config as X_cfg

from tensorflow.keras.applications import EfficientNetB0, EfficientNetB2, EfficientNetB4, MobileNetV3Large, Xception, DenseNet169, VGG19

from keras.layers import Input
from keras import Model
from keras.optimizers import gradient_descent_v2
from tensorflow.keras.optimizers import Adam, RMSprop

BASE_DIR = Path('test_neurals.py').resolve(strict=True).parent

shape = (224, 224)
train_gen, valid_gen = train.datagen(img_size=shape)
opt = gradient_descent_v2.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
opt2 = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt3 = RMSprop(learning_rate=1e-4, epsilon=1e-08)
EPOCHS = 15

# 1. XResNet50
#XResNet = XRN50().build(input_shape = (cfg.IMAGE_W, cfg.IMAGE_H, 3), num_classes = cfg.NUM_CLASSES, summary = True)
#train.prepare_model(XResNet, "XRN50_best.h5")
#XResNet.compile(loss='categorical_crossentropy', optimizer=opt2,metrics=['accuracy'])
#H = XResNet.fit(train_gen,validation_data=valid_gen,epochs=cfg.EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(cfg.CB_NAME, es=False))

# 2. EfficientNet
#inputs = Input(shape=(EN_cfg.IMAGE_W, EN_cfg.IMAGE_H, 3))
#outputs = EfficientNetB0(include_top=True, weights=None, classes=EN_cfg.NUM_CLASSES)(inputs)
#EfficientNet = Model(inputs, outputs)
#EfficientNet.summary()
#EfficientNet.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
#H2 = EfficientNet.fit(train_gen,validation_data=valid_gen,epochs=EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(EN_cfg.CB_NAME, es=False))
#plot_hist(H2, filename = BASE_DIR.joinpath('test_results', 'ENB0-20EP-SGD(lr-4)-BATCH16.png'))
'''
inputs = Input(shape=(260, 260, 3))
outputs = EfficientNetB2(include_top=True, weights=None, classes=EN_cfg.NUM_CLASSES)(inputs)
EfficientNet = Model(inputs, outputs)
EfficientNet.summary()
EfficientNet.compile(loss='categorical_crossentropy', optimizer=opt2,metrics=['accuracy'])
H = EfficientNet.fit(train_gen,validation_data=valid_gen,epochs=EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(EN_cfg.CB_NAME, es=False))
plot_hist(H, filename = BASE_DIR.joinpath('test_results', 'ENB2-20EP-Adam(lr-5)-BATCH16.png'))
'''
#inputs = Input(shape=(EN4_cfg.IMAGE_W, EN4_cfg.IMAGE_H, 3))
#outputs = EfficientNetB4(include_top=True, weights=None, classes=EN4_cfg.NUM_CLASSES)(inputs)
#EfficientNet = Model(inputs, outputs)
#EfficientNet.compile(loss='categorical_crossentropy', optimizer=opt2,metrics=['accuracy'])
#H = EfficientNet.fit(train_gen,validation_data=valid_gen,epochs=EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(EN4_cfg.CB_NAME, es=False))
#plot_hist(H, filename = BASE_DIR.joinpath('test_results', 'ENB4-15EP-Adam(lr-4)-BATCH16.png'))
'''
# 3. MobileNetV3
inputs = Input(shape=(M3L_cfg.IMAGE_W, M3L_cfg.IMAGE_H, 3))
outputs = MobileNetV3Large(include_top=True, weights=None, classes=M3L_cfg.NUM_CLASSES,
                            alpha=1.0, dropout_rate=0.05)(inputs)
MobileNetV3Net = Model(inputs, outputs)
MobileNetV3Net.compile(loss='categorical_crossentropy', optimizer=opt2,metrics=['accuracy'])
H3 = MobileNetV3Net.fit(train_gen,validation_data=valid_gen,epochs=EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(M3L_cfg.CB_NAME, es=False))
plot_hist(H3, filename = BASE_DIR.joinpath('test_results', 'M3L-15EP-Adam(lr-5)-BATCH16.png'))
'''
# 4. DenseNet
inputs = Input(shape=(Dense_cfg.IMAGE_W, Dense_cfg.IMAGE_H, 3))
outputs = DenseNet169(include_top=True, weights=None, classes=Dense_cfg.NUM_CLASSES)(inputs)
DenseNet = Model(inputs, outputs)
DenseNet.summary()
DenseNet.compile(loss='categorical_crossentropy', optimizer=opt2,metrics=['accuracy'])
H4 = DenseNet.fit(train_gen,validation_data=valid_gen,epochs=EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(Dense_cfg.CB_NAME, es=False))
plot_hist(H4, filename = BASE_DIR.joinpath('test_results', 'Dense169-15EP-Adam(lr-4)-BATCH16.png'))
'''
# 5. VGG19
inputs = Input(shape=(VGG19_cfg.IMAGE_W, VGG19_cfg.IMAGE_H, 3)) 
outputs = VGG19(include_top=True, weights=None, classes=VGG19_cfg.NUM_CLASSES)(inputs)
VGG19Net = Model(inputs, outputs)
VGG19Net.summary()
VGG19Net.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
H5 = VGG19Net.fit(train_gen,validation_data=valid_gen,epochs=EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(VGG19_cfg.CB_NAME, es=False))
plot_hist(H5, filename = BASE_DIR.joinpath('test_results', 'VGG19-15EP-Adam(lr-4)-BATCH16.png'))

# 6. Xception
inputs = Input(shape=(X_cfg.IMAGE_W, X_cfg.IMAGE_H, 3)) 
outputs = Xception(include_top=True, weights=None, classes=X_cfg.NUM_CLASSES)(inputs)
XceptionNet = Model(inputs, outputs)
XceptionNet.compile(loss='categorical_crossentropy', optimizer=opt ,metrics=['accuracy'])
H6 = XceptionNet.fit(train_gen,validation_data=valid_gen,epochs=EPOCHS,verbose=1,callbacks=callbacks.get_callbacks(X_cfg.CB_NAME, es=False))
plot_hist(H6, filename = BASE_DIR.joinpath('test_results', 'X-15EP-Adam(lr-4)-BATCH16.png'))
''' 