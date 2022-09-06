##############################################################
# segmentation (Unet) data preprocessing class               #
# 13.04.2022                                                 #
# v2.2 (17.04.2022)                                          #
##############################################################

import cv2
import numpy as np
import os
import random

from keras.preprocessing.image import ImageDataGenerator as IDG

class Img_preprocesser:
    def __init__(self):
        pass

    def imageToOHE_old(self,image,cn=3):
        # one hot encoding (mask)
        # create zeros array > fill channels (3) with 1 and 0
        # used colors-divided classes: 
        # background: red(200-255, 0, 0)
        # working area: green(0, 200-255, 0)
        # working piece: yellow(200-255, 200-255, 0)
        # returns H*W*Classes one-hot encoded mask
        pic = image
        res = np.zeros((pic.shape[0], pic.shape[1], cn))
        for x in range(pic.shape[0]):
            for y in range(pic.shape[1]):
                if pic[x][y][0] > 200 and pic[x][y][1] > 200:
                    res[x][y][2] = 1
                elif pic[x][y][1] > 200 and pic[x][y][0] < 200:
                    res[x][y][1] = 1
                else:
                    res[x][y][0] = 1
        return res

    def imageToOHE(self,image):
        # one hot encoding (mask)
        # create zeros array > fill channels (3) with 1 and 0
        # used colors-divided classes: 
        # background: red(200-255, 0, 0)
        # working area: green(0, 200-255, 0)
        # working piece: yellow(200-255, 200-255, 0)
        # returns H*W*Classes one-hot encoded mask
        OHE = np.zeros_like(image)
        OHE[:, :, 2][np.logical_and(image[:, :, 0] > 200, image[:, :, 1] > 200)] = 1
        OHE[:, :, 1][np.logical_and(image[:, :, 0] < 200, image[:, :, 1] > 200)] = 1
        OHE[:, :, 0][np.logical_and(image[:, :, 0] > 200, image[:, :, 1] < 200)] = 1
        
        return OHE

    def OHEtoImage(self, encoded):
        mask = np.zeros_like(encoded)
        mask[:, :, 0] = 255
        mask[:, :, 0:3][encoded[:, :, 1] == 1] = (0, 255, 0)
        mask[:, :, 0:3][encoded[:, :, 2] == 1] = (255, 240, 0)
        
        return mask  

    def load_dataset(self, img_path, mask_path, size):
        # list all images and masks > read image > change color scheme to rgb > resize
        # for image: append to X > transform to array > normalize
        # for mask: OHE > append to Y > transform to array
        # returns X, Y arrays
        X = []
        Y = []

        images = os.listdir(img_path)
        masks = os.listdir(mask_path)

        for image in images:
            path = os.path.join(img_path, image)
            image_ = cv2.imread(path)
            image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
            image_ = cv2.resize(image_, size)
            X += [image_]

        for mask in masks:
            path = os.path.join(mask_path, mask)    
            mask_ = cv2.imread(path)
            mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2RGB)
            mask_ = cv2.resize(mask_, size, cv2.INTER_NEAREST)
            mask_ = self.imageToOHE(mask_)
            Y += [mask_]

        X = np.array(X) / 255.
        Y = np.array(Y)
        return X,Y

    def aug_data(self, X, Y, frequency=0.2):
        # augment data with different transforms with {frequency}
        # take X[i] and Y[i] > apply same transform to both  > append to X_aug, Y_aug >
        # > vertical stack augmented array with original array > return X, Y
        len = X.shape[0]

        angles = [90, 180, 270]
        X_aug = []
        Y_aug = []

        for i in range(0, len, int(1/frequency)):
            angle = random.choice(angles)
            X_aug += [IDG().apply_transform(X[i], {'theta': angle})]
            Y_aug += [IDG().apply_transform(Y[i], {'theta': angle})]
        
        for i in range(0, len, int(1/frequency)):
            X_aug += [IDG().apply_transform(X[i], {'flip_horizontal': True})]
            Y_aug += [IDG().apply_transform(Y[i], {'flip_horizontal': True})]

        for i in range(0, len, int(1/frequency)):
            X_aug += [IDG().apply_transform(X[i], {'flip_vertical': True})]
            Y_aug += [IDG().apply_transform(Y[i], {'flip_vertical': True})]

        X = np.vstack([X, np.array(X_aug)])

        Y = np.vstack([Y, np.array(Y_aug)])

        return X, Y
