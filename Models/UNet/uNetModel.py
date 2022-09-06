##########################################################################
# UNet model class                                                       #
# contains encoder, decoder (and blocks)                                 #
# use build_UNet(shape() to construct whole model                        #
# 10.04.2022                                                             #
# v1.1 (11.04.2022)                                                                   #
##########################################################################

import cv2
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Activation, Flatten, Dropout, Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
import os

import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator as IDG
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

from Models.UNet import config as cfg
from Lib import callbacks 
from Lib.metrics import dice_similarity_coefficient as DSC
from Lib.load_seg_data import Img_preprocesser as ipre

class UNet:
    def __init__(self, filters = 16, bn = True, num_classes = 2):
        self.filters = filters
        self.bn = bn
        self.num_classes = num_classes

    #############################################################################
    #                                    BUILD                                  #
    #############################################################################

    def encoder_block(self, X_Input, filters, dropout):
        X = Conv2D(filters, (3, 3), padding='same')(X_Input)
        if self.bn: X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters, (3, 3), padding='same')(X)
        if self.bn: X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = MaxPooling2D(pool_size=(2,2))(X)
        X = Dropout(dropout)(X)

        return X

    def decoder_block(self, X_Input, filters, dropout):
        X = Conv2D(filters, (3, 3), padding='same')(X_Input)
        if self.bn: X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters, (3, 3), padding='same')(X)
        if self.bn: X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Dropout(dropout)(X)

        return X

    def encode(self, X_Input, filters):
        conv1 = self.encoder_block(X_Input, filters, dropout=0.1)
        conv2 = self.encoder_block(conv1, filters*2, dropout=0.1)
        conv3 = self.encoder_block(conv2, filters*4, dropout=0.1)
        conv4 = self.encoder_block(conv3, filters*8, dropout=0.1)

        return conv1, conv2, conv3, conv4

    def decode(self, conv1, conv2, conv3, conv4, conv5, filters):
        upsample = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(conv5)
        concat = concatenate([conv4, upsample])
        X = self.decoder_block(concat, filters, dropout=0.1)
        
        upsample = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(X)
        concat = concatenate([conv3, upsample])
        X = self.decoder_block(concat, filters//2, dropout=0.1)

        upsample = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(X)
        concat = concatenate([conv2, upsample])
        X = self.decoder_block(concat, filters//4, dropout=0.1)

        upsample = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(X)
        concat = concatenate([conv1, upsample])
        X = self.decoder_block(concat, filters//8, dropout=0.1)

        return X

    def build_UNet(self, shape = (cfg.IMAGE_W, cfg.IMAGE_H, 3)):
        self.current_shape = shape
        X_Input = Input(shape=self.current_shape)

        conv1, conv2, conv3, conv4 = self.encode(X_Input, self.filters)

        conv5 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(self.filters*16, (3, 3), padding='same')(conv5)
        if self.bn: conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)

        conv5 = Conv2D(self.filters*16, (3, 3), padding='same')(conv5)
        if self.bn: conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)

        decoded = self.decode(conv1, conv2, conv3, conv4, conv5, self.filters*8)

        upsample = Conv2DTranspose(self.filters, (3, 3), strides=(2, 2), padding='same')(decoded)

        output = Conv2D(self.num_classes, (1, 1), activation='sigmoid', padding='same')(upsample)

        self.model = Model(X_Input, output)
        #return model

    #############################################################################
    #                                    TRAIN                                  #
    #############################################################################

    def save_model(self, save_to = cfg.SAVE_PATH, model_path = cfg.MODEL_PATH):
        # convert model to json, make path dirs if not exists, save
        model_json = self.model.to_json()
        if not os.path.isdir(save_to): os.makedirs(save_to)
        with open(model_path,"w") as json_file:
            json_file.write(model_json)

    def train_model(self, data_summary = True, image_path = cfg.IMAGE_PATH, mask_path = cfg.MASK_PATH, img_size = (cfg.IMAGE_W, cfg.IMAGE_H), epochs = cfg.EPOCHS, save = True):
        X, Y = ipre().load_dataset(image_path, mask_path, img_size)
                
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
        # is gpu aviable
        print(device_lib.list_local_devices())
        
        # set optimizer, compile model, train with generator
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer = adam, loss='binary_crossentropy', metrics=['accuracy', DSC])
        H = self.model.fit_generator(train_gen,validation_data=valid_gen,epochs=epochs,verbose=1,callbacks=callbacks.get_callbacks(cfg.CB_NAME, es=False))

        # save model if flagged
        if save == True: self.save_model()
        return H

    def prepare_model(self, weights = "UNet_best.h5"):
        # load model weights
        self.model.load_weights(weights)

    #############################################################################
    #                                    TEST                                   #
    #############################################################################

    def get_mask(self, image):
        # read image > resize it > predict > return results 
        encoded_mask = np.around(self.model.predict(np.expand_dims(image, axis=0))[0], 0)
        result = ipre().OHEtoImage(encoded_mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        with open('mask_file.txt', 'w') as f:
            csv.writer(f, delimiter=' ').writerows(encoded_mask)

        return result