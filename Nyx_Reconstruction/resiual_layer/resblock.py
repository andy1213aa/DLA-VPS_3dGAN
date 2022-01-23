import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa
class ResBlock_generator(layers.Layer):
  def __init__(self, out_shape, strides=1, ksize = 3, shortcut = False):
    super(ResBlock_generator, self).__init__()
    self.shortcut = shortcut
    
    # self.upSample = layers.UpSampling3D()
    self.conv_0 = layers.Conv3DTranspose(out_shape,kernel_size = ksize, strides=2,padding='same', name = 'rg_conv1',  use_bias=False)
    self.bn_0 = layers.BatchNormalization()
    self.PRelu0 = layers.LeakyReLU(name='G_LeakyReLU1')
    self.conv_1 =layers.Conv3D(out_shape,kernel_size = ksize ,strides=1,padding='same', name = 'rg_conv2', use_bias=False)
    self.bn_1 = layers.BatchNormalization()
    self.PRelu1 = layers.LeakyReLU(name='G_LeakyReLU2')
    self.conv_2 =layers.Conv3D(out_shape,kernel_size = ksize ,strides=1,padding='same', name = 'rg_conv3', use_bias=False)
    self.bn_2 = layers.BatchNormalization()
    self.PRelu2 = layers.LeakyReLU(name='G_LeakyReLU3')
    self.conv_3 =layers.Conv3D(out_shape,kernel_size = ksize ,strides=1,padding='same', name = 'rg_conv4', use_bias=False)
    
    self.bn_3 = layers.BatchNormalization()
    

    if shortcut:
      # self.upSample_shortcut = layers.UpSampling3D()
      self.conv_shortcut = layers.Conv3DTranspose(out_shape,kernel_size=1,strides=2, padding='same', use_bias=False)
      

    self.PRelu3 = layers.LeakyReLU(name='G_LeakyReLU4')
  def call(self, inputs):
    # x = self.upSample(inputs)
    x = self.conv_0(inputs)
    x = self.bn_0(x)
    x = self.PRelu0(x)
    x = self.conv_1(x)
    x = self.bn_1(x)
    x = self.PRelu1(x)
    x = self.conv_2(x)
    x = self.bn_2(x)
    x = self.PRelu2(x)
    x = self.conv_3(x)
    x = self.bn_3(x)
    
    
    if self.shortcut:
      # shortcut = self.upSample_shortcut(inputs)
      shortcut = self.conv_shortcut(inputs)
      x = layers.add([x,shortcut])

    outputs = self.PRelu3(x)
    return outputs

class ResBlock_discriminator(layers.Layer):
  def __init__(self, out_shape, strides=1, ksize=3, shortcut = False):
    super(ResBlock_discriminator, self).__init__()
    self.shortcut = shortcut
    self.conv_0 = tfa.layers.SpectralNormalization(layers.Conv3D(out_shape,kernel_size=ksize,strides=2,padding='same', name = 'rd_conv1', use_bias=False))
    self.PRelu0 = layers.LeakyReLU(name='D_LeakyReLU0')

    #shortcut
    if shortcut:
      self.conv_shortcut = tfa.layers.SpectralNormalization(layers.Conv3D(out_shape, kernel_size=1 ,strides=2, padding='valid', use_bias=False))

    # self.PRelu2 = layers.LeakyReLU(name='D_LeakyReLU2')
  def call(self, inputs):

    x = self.conv_0(inputs)

    if self.shortcut:
      shortcut = self.conv_shortcut(inputs)
      x = layers.add([x,shortcut])

    output = self.PRelu0(x)
    return output

