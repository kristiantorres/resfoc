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
  u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
  u = LeakyReLU(alpha=0.2)(u)
  if(dropout):
    u = Dropout(dropout)(u)
  if(bn):
    u = BatchNormalization(momentum=0.8)(u)
  if(skip_input not None):
    u = Concatenate()([u, skip_input])
  return u

def conv2dt(layer, filters, f_size, skip_input=None, bn=True, dropout=0.0):
  """ A wrapper for Keras transposed convolution """
  u = Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same')(u)
  u = LeakyReLU(alpha=0.2)(u)
  if(dropout):
    u = Dropout(dropout)(u)
  if(bn):
    u = BatchNormalization(momentum=0.8)(u)
  if(skip_input not None):
    u = Concatenate()([u, skip_input])
  return u

def auto_encoder(imgshape, nc_in, nc_out, nf, ksize, unet=True, dropout=0.0):
  """ Returns a Keras autoencoder model. By default gives a u-net
  with skip connections """

  inp = Input(shape(imgshape[0],imgshape[1],imgshape[3]), name='input')

  # Encoder
  d1 = conv2d(inp, nf, ksize, bn=False)
  d2 = conv2d(d1, nf *  2, ksize)
  d3 = conv2d(d2, nf *  4, ksize)
  d4 = conv2d(d3, nf *  8, ksize)
  d5 = conv2d(d4, nf * 16, ksize)

  # Decoder
  u5 = None
  if(unet):
    u1 = deconv2d(d5, nf * 16, ksize, d4)
    u2 = deconv2d(u1, nf *  8, ksize, d3)
    u3 = deconv2d(u2, nf *  4, ksize, d2)
    u4 = deconv2d(u3, nf *  2, ksize, d1)
    u5 = deconv2d(u4, nf *  1, ksize)
  else:
    u1 = deconv2d(d5, nf * 16, ksize)
    u2 = deconv2d(u1, nf *  8, ksize)
    u3 = deconv2d(u2, nf *  4, ksize)
    u4 = deconv2d(u3, nf *  2, ksize)
    u5 = deconv2d(u4, nf *  1, ksize)

  out = Conv2D(nc_in, kernel_size=ksize, strides=1, padding='same', activation='relu', name='output')(u5)

  return Model(inp,out)

