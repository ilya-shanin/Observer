##############################################################
# Config for xResNet50                                       #
# 12.04.2022                                                 #
# v1.0                                                       #
##############################################################

import os
from Lib import callbacks

CB_NAME = 'EfficientNetB0'
MODEL_NAME = 'EfficientNetB0'
MODELS_DIR = 'Models'
TRAIN_PATH = os.path.join('Data', 'Train')
VALID_PATH = os.path.join('Data', 'Valid')
SAVE_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_PATH = os.path.join(SAVE_PATH, 'model.json')

EPOCHS = 15
IMAGE_H = 380
IMAGE_W = 380
BATCH_SIZE = 16
NUM_CLASSES = 4
CALLBACKS = callbacks.get_callbacks(CB_NAME, es=True)