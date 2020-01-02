# Functions for building deep neural networks using Keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, 
     BatchNormalization, Concatenate, Activation, Cropping2D,
     ZeroPadding2D, UpSampling2D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

def conv2d(layer_input, filters, f_size, bn=True, dropout=0.0):
  """ A wrapper for Keras Conv2D function """
  d = Conv2d(filters, kernel_size=f_size, strides=2, padding='same')
  d = LeakyReLU(alpha=0.2)(d)
  if(dropout):
    d = Dropout(dropout)(d)
  if(bn):
    d = BatchNormalization(momentum=0.8)(d)
  return d

def deconv2d(layer, filters, f_size, skip_input=None, bn=True, dropout=0.0):
  """ A wrapper for Keras upsampled convolution """
  u = UpSampling2D(size=2)(layer_input)
  u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
  if(dropout):
    u = Dropout(dropout)(u)
  if(bn):
    u = BatchNormalization(momentum=0.8)(u)
  if(skip_input not None):
    u = Concatenate()([u, skip_input])
  return u

def auto_encoder(imgshape, nc_in, nc_out, ksize, unet=True, dropout=0.0):
  """ Returns a Keras autoencoder model. By default gives a u-net
  with skip connections """


