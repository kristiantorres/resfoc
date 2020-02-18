"""
Custom loss functions for training Keras nets

@author: Joseph Jennings
@version: 2020.02.17
"""
import tensorflow as tf
import tensorflow.keras.backend as K

def wgtbce(weight=0.5):

  def mywgtbce(y_true,y_pred):
    b_ce = K.binary_crossentropy(y_true, y_pred)

    # Apply the weights
    weight_vector = y_true * weight + (1. - y_true) * (1 - weight)
    wbce = weight_vector * b_ce

    # Return the mean error
    return K.mean(wbce)

  return mywgtbce
