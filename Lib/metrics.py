##############################################################
# dice similarity coefficient                                #
# 16.04.2022                                                 #
# v1.1 (17.04.2022)                                          #
##############################################################

from keras import backend as K

def dice_similarity_coefficient(y_true, y_pred):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    summation = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2.0 * intersection + K.epsilon()) / (summation + K.epsilon())