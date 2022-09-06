##############################################################
# Config for Xception                                        #
# 12.04.2022                                                 #
# v1.0                                                       #
##############################################################

import os
from Lib import callbacks

CB_NAME = 'Xception'
MODEL_NAME = 'Xception'
MODELS_DIR = 'Models'
TRAIN_PATH = os.path.join('Data', 'Train')
VALID_PATH = os.path.join('Data', 'Valid')
SAVE_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_PATH = os.path.join(SAVE_PATH, 'model.json')

EPOCHS = 10
IMAGE_H = 299
IMAGE_W = 299
BATCH_SIZE = 8
NUM_CLASSES = 4
CALLBACKS = callbacks.get_callbacks(CB_NAME, es=True)