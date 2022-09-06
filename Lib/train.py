##############################################################
# loading data, training model, saving model, loading weigts #
# 16.04.2020                                                 #
# v1.0                                                       #
##############################################################

import os
from keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.python.client import device_lib
from keras.optimizers import gradient_descent_v2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

from Lib.load_seg_data import Img_preprocesser as ipre
from Models.XRN50.XRN50_Model import XRN50
from Models.XRN50 import config as cfg
from Models.UNet import config as cfgU

def save_model(model, save_to = cfg.SAVE_PATH, model_path = cfg.MODEL_PATH):
    # convert model to json, make path dirs if not exists, save
    model_json = model.to_json()
    if not os.path.isdir(save_to): os.makedirs(save_to)
    with open(model_path,"w") as json_file:
        json_file.write(model_json)

def datagen(train_path = cfg.TRAIN_PATH, valid_path = cfg.VALID_PATH, batch = cfg.BATCH_SIZE, img_size = (cfg.IMAGE_W, cfg.IMAGE_H)):
    # get classes by dirs
    class_names=os.listdir(train_path) 
    class_names_valid=os.listdir(valid_path)
    print(class_names)
    print(class_names_valid)

    # normalize and random augment data
    train_datagen = IDG(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15, rescale=1.0/255.0) 
    valid_datagen = IDG(rescale=1.0/255.0)

    # take data from dirs
    train_gen = train_datagen.flow_from_directory(train_path,target_size=img_size,batch_size=batch,shuffle=True,class_mode='categorical')
    valid_gen = valid_datagen.flow_from_directory(valid_path,target_size=img_size,batch_size=batch,shuffle=True,class_mode='categorical')
    return train_gen, valid_gen

def segmentation_datagen(shape = (cfgU.IMAGE_W, cfgU.IMAGE_H), image_path = cfgU.IMAGE_PATH, mask_path = cfgU.MASK_PATH, data_summary = True):
    # defining data generator for image segmentation
    X, Y = ipre().load_dataset(image_path, mask_path, shape)

    #X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42) # test split not nescessary now
                
    if data_summary:
        print('X: ', len(X), 'Y: ', len(Y))
        print(np.shape(X),np.shape(Y))

    X, Y = ipre().aug_data(X, Y)

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)

    if data_summary:
            print('X augmented: ', len(X), 'Y augmented: ', len(Y))
            print('Train: ', len(X_train),'/', len(Y_train), '\nValidation: ', len(X_valid),'/', len(Y_valid))
        
    train_gen = IDG().flow(X_train, Y_train, batch_size=cfg.BATCH_SIZE)
    valid_gen = IDG().flow(X_valid, Y_valid, batch_size=cfg.BATCH_SIZE)
    return train_gen, valid_gen

def train_model(model, train_gen, valid_gen, optimizer = "SGD", metrics = ["accuracy"], loss="categorical_crossentropy", 
                epochs = cfg.EPOCHS, callbacks = cfg.CALLBACKS, EarlyStop = False, save = True):

    # is gpu aviable
    print(device_lib.list_local_devices())
    
    # set optimizer, compile model, train with generator
    if optimizer == "SGD": opt = gradient_descent_v2.SGD(learning_rate=1e-4, momentum=0.9) # for XRN50 
    elif optimizer == "Adam": opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #for Unet
    model.compile(loss=loss, optimizer=opt,metrics=metrics)
    H = model.fit_generator(train_gen,validation_data=valid_gen,epochs=epochs,verbose=1,callbacks=callbacks.get_callbacks(cfg.CB_NAME, es=EarlyStop))

    # save model if flagged
    if save == True: save_model(model)
    return H

def prepare_model(model, weights = "XRN50_best.h5"):
    # load model weights
    model.load_weights(weights)

#############################################################
################## test train_model #########################
#############################################################
def test():
    Train, Valid = datagen()
    model = XRN50().build(input_shape = (cfg.IMAGE_W, cfg.IMAGE_H, 3), num_classes = cfg.NUM_CLASSES, summary = False)
    train_model(model, Train, Valid)