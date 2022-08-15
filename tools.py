import keras.backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

def gdice_coef(y_true, y_pred):
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    A = K.sum(y_true, axis=[1, 2, 3])
    B = K.sum(y_pred, axis=[1, 2, 3])

    TP = K.sum(y_true * y_pred, axis=[1, 2, 3])
    two_TP = 2. * TP

    FN = A - TP
    FP = B - TP
    FP_1 = FP + (K.square(FP) / (TP + FN + K.epsilon()))

    G_DICE = (two_TP + K.epsilon()) / (two_TP + FN + FP_1 + K.epsilon())
    return (G_DICE)


def gdice_coef_loss(y_true, y_pred):
    return -gdice_coef(y_true, y_pred)


def sgdice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    A = K.sum(y_true_f, axis=[1, 2, 3])
    B = K.sum(y_pred_f, axis=[1, 2, 3])

    TP = K.sum(y_true_f * y_pred_f, axis=[1, 2, 3])
    two_TP = 2. * TP

    FN = A - TP
    FP = B - TP
    FP_1 = FP + (K.square(FP) / (TP + FN - FP + K.epsilon()))

    SG_DICE = (two_TP + K.epsilon()) / (two_TP + FN + FP_1 + K.epsilon())
    return (SG_DICE)


def sgdice_coef_loss(y_true, y_pred):
    return -sgdice_coef(y_true, y_pred)


def IoU(y_true, y_pred, eps=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)


def zero_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)

def agg_loss(in_gt, in_pred):
    return -1e-2 * zero_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)
