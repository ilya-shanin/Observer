####################################################
# callbacks: model checkpoint (MC), early stopping #
# MC saves last & best weights                     #
# 13.04.2022                                       #
# v1.0                                             #
####################################################

from keras.callbacks import EarlyStopping,ModelCheckpoint

def get_callbacks(prefix, es=False):
    callbacks = []
    if es: 
        ES = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
        callbacks.append(ES)

    best_w = ModelCheckpoint(prefix + '_best.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

    last_w = ModelCheckpoint(prefix + '_last.h5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=True,
                                    mode='auto',
                                    period=1)

    callbacks.append(best_w)
    callbacks.append(last_w)
    return callbacks