##########################################################################
# xResNet50 model class                                                  #
# contains typical XRN50 blocks                                          #
# use build(input_shape, num_classes, summary) to construct whole model  #
# 13.04.2022                                                             #
# v1.0                                                                   #
##########################################################################

from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.initializers import glorot_uniform

class XRN50:
    def __init__(self):
        pass

    def input_block(self, X, f, filters, stage, block, s=2):
        layer_name_base = 'inp' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        pool_name = 'maxpool' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(s, s), padding='same', name=layer_name_base + '1a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '1a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=layer_name_base + '1b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '1b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=(f, f), strides=(1, 1), padding='same', name=layer_name_base + '1c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '1c')(X)
        X = Activation('relu')(X)

        X = MaxPooling2D((3, 3), strides=(2, 2), name=pool_name + '1d')(X)
        return X

    def identity_block(self, X, f, filters, stage, block):
   
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters

        X_shortcut = X
    
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X = Add()([X, X_shortcut])# SKIP Connection
        X = Activation('relu')(X)

        return X

    def xdownsampling_block(self, X, f, filters, stage, block, s=2):
   
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        pool_name = 'avgpool' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        X_shortcut = X

        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(2, 2), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '1a', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name=pool_name + '1b')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def build_encoder(self, input_shape):

        X_input = Input(input_shape)

        X = ZeroPadding2D((1, 1))(X_input)

        X = self.input_block(X, 3, [32, 64, 64], stage=1, block='a', s=2)

        X = self.xdownsampling_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')


        X = self.xdownsampling_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        X = self.xdownsampling_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        X = self.xdownsampling_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
        
        model = Model(inputs=X_input, outputs=X)

        return model

    def build(self, input_shape=(224, 224, 3), num_classes=2, summary = True):
        base_model = self.build_encoder(input_shape=input_shape)
        headModel = base_model.output
        headModel = Flatten()(headModel)
        headModel= Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
        headModel= Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
        headModel = Dense(num_classes, activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)

        model = Model(inputs=base_model.input, outputs=headModel, name='XResNet50')
        if summary: model.summary()
        return model