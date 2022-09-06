##############################################################
# Config for UNet                                            #
# 04.04.2022                                                 #
# v1.0                                                       #
##############################################################

import os
from Lib import callbacks

CB_NAME = 'UNet'
MODEL_NAME = 'UNet'
MODELS_DIR = 'Models'
IMAGE_PATH = os.path.join('Data', 'seg_image')
MASK_PATH = os.path.join('Data', 'seg_mask')
SAVE_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_PATH = os.path.join(SAVE_PATH, 'UNet.json')

EPOCHS = 20
IMAGE_H = 256
IMAGE_W = 256
BATCH_SIZE = 3
NUM_CLASSES = 3
CALLBACKS = callbacks.get_callbacks(CB_NAME, es=False)

