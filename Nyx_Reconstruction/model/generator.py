from ..resiual_layer.resblock import ResBlock_generator
import tensorflow as tf
from tensorflow.keras import layers, initializers

from IPython.display import Image


class generator(tf.keras.Model):
    def __init__(self, ch= 32):
        super(generator, self).__init__()
        self.ch = ch
        # self.inputBN = layers.BatchNormalization()
        self.xD0 = layers.Dense(512, name = 'parameter1_layer_1')
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        # self.xD0BN = layers.BatchNormalization()
        self.xR0 = layers.LeakyReLU()
        self.xD1 = layers.Dense(512, name = 'parameter1_layer_2')
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        # self.xD1BN = layers.BatchNormalization()
        self.xR1 = layers.LeakyReLU()
        self.xD2 = layers.Dense(512, name = 'parameter1_layer_3')
        # if self.hparams[HP_BN_UNITS] : x = layers.BatchNormalization()(x)
        # self.xD2BN = layers.BatchNormalization()
        self.xR2 = layers.LeakyReLU()
        
        self.yD0 = layers.Dense(512, name = 'parameter2_layer_1')
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        # self.yD0BN = layers.BatchNormalization()
        self.yR0 = layers.LeakyReLU()
        self.yD1 = layers.Dense(512, name = 'parameter2_layer_2')
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        # self.yD1BN = layers.BatchNormalization()
        self.yR1 = layers.LeakyReLU()
        self.yD2 = layers.Dense(512, name = 'parameter2_layer_3')
        # if self.hparams[HP_BN_UNITS] : y = layers.BatchNormalization()(y)
        # self.yD2BN = layers.BatchNormalization()
        self.yR2 = layers.LeakyReLU()

        
        self.zD0 = layers.Dense(512, name = 'parameter3_layer_1')
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        # self.zD0BN = layers.BatchNormalization()
        self.zR0 = layers.LeakyReLU()
        self.zD1 = layers.Dense(512, name = 'parameter3_layer_2')
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        # self.zD1BN = layers.BatchNormalization()
        self.zR1 = layers.LeakyReLU()
        self.zD2 = layers.Dense(512, name = 'parameter3_layer_3')
        # if self.hparams[HP_BN_UNITS] : z = layers.BatchNormalization()(z)
        # self.zD2BN = layers.BatchNormalization()
        self.zR2 = layers.LeakyReLU()

        
        self.gD0 =layers.Dense(ch*2*4*4*4)
      
        self.gRes2 = ResBlock_generator(8*ch, shortcut=True)
        self.gRes3 = ResBlock_generator(4*ch, shortcut=True)
        self.gRes4 = ResBlock_generator(2*ch, shortcut=True)
        #self.gR0 = layers.LeakyReLU()
        self.gRes5 = ResBlock_generator(ch, shortcut=True)
        #g = layers.Conv3DTranspose((2**i)*self.filterNumber, kernel_size=3, strides=2, padding='same', use_bias=False)(g)
        

        #g = SpectralNormalization(layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', use_bias=False))(g)
        self.gConv0 = layers.Conv3D(1, kernel_size=3, strides=1, padding='same', use_bias=False)
        
        # self.gOutput = layers.ReLU()
        self.gOutput = layers.Activation('tanh')
        #plot_model(model, to_file='WGAN_generator.png', show_shapes=True)
    
    def call(self, inputs, training = True):

        # p1, p2, p3 = inputs
        # x = self.inputBN(inputs)
        x = self.xD0(inputs)
        # x = self.xD0BN(x)
        x = self.xR0(x)
        x = self.xD1(x)
        # x = self.xD1BN(x)
        x = self.xR1(x)
        x = self.xD2(x)
        # x = self.xD2BN(x)
        x = self.xR2(x)

        y = self.yD0(x)
        # y = self.yD0BN(y)
        y = self.yR0(y)
        y = self.yD1(y)
        # y = self.yD1BN(y)
        y = self.yR1(y)
        y = self.yD2(y)
        # y = self.yD2BN(y)
        y = self.yR2(y)

        z = self.zD0(y)
        # z = self.zD0BN(z)
        z = self.zR0(z)
        z = self.zD1(z)
        # z = self.zD1BN(z)
        z = self.zR1(z)
        z = self.zD2(z)
        # z = self.zD2BN(z)
        z = self.zR2(z)

        # xyz = layers.concatenate([x, y, z])
        xyz = self.gD0(z)
        xyz = layers.LeakyReLU()(xyz)
        xyz = layers.Reshape((4, 4, 4, 2*self.ch))(xyz)

        g = self.gRes2(xyz, training = training)
        g = self.gRes3(g, training = training)
        g = self.gRes4(g, training = training)
        g = self.gRes5(g, training = training)
        g = self.gConv0(g)
        g =  self.gOutput(g)
        return g
    def model(self):
        parameterInput = tf.keras.Input(shape=(1, 3), name='parameter')

        return tf.keras.models.Model(inputs=parameterInput, outputs = self.call(parameterInput))