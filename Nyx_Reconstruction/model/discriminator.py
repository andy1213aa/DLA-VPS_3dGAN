from ..resiual_layer.resblock import ResBlock_discriminator
import tensorflow as tf
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa



class discriminator(tf.keras.Model):
    def __init__(self, ch= 32):
        super(discriminator, self).__init__()
        self.ch = ch
       
        self.xD0 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter1_layer_1'))

        self.xR0 = layers.LeakyReLU()
        self.xD1 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter1_layer_2'))
       
        self.xR1 = layers.LeakyReLU()
        self.xD2 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter1_layer_3'))
    
        self.xR2 = layers.LeakyReLU()
        
        self.yD0 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter2_layer_1'))

        self.yR0 = layers.LeakyReLU()
        self.yD1 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter2_layer_2'))
       
        self.yR1 = layers.LeakyReLU()
        self.yD2 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter2_layer_3'))

        self.yR2 = layers.LeakyReLU()

        
        self.zD0 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter3_layer_1'))

        self.zR0 = layers.LeakyReLU()
        self.zD1 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter3_layer_2'))
        
        self.zR1 = layers.LeakyReLU()
        self.zD2 = tfa.layers.SpectralNormalization(layers.Dense(512, name = 'parameter3_layer_3'))
      
        self.zR2 = layers.LeakyReLU()

     
        self.res0 = ResBlock_discriminator(ch*2, ksize=4, shortcut=True)
        self.res1 = ResBlock_discriminator(ch*4, ksize=4 , shortcut=True)
        self.res2 = ResBlock_discriminator(ch*8, ksize=4, shortcut=True)
        self.res3 = ResBlock_discriminator(ch*16, ksize=4, shortcut=True)

        self.GAV3D = layers.GlobalAveragePooling3D()
        
        self.multiple = layers.Multiply()
        self.sum = tf.math.reduce_sum
        
        self.out = layers.add
        self.outputD = tfa.layers.SpectralNormalization(layers.Dense(1))
        self.outputActivation = layers.Activation('linear')
     
    def call(self, inputs, training=None):
        p, data = inputs
        # x = self.inputBN(p)
        x = self.xD0(p)
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



        d = self.res0(data)
        d = self.res1(d)
        d = self.res2(d)
        d = self.res3(d)


        d = self.GAV3D(d)
  
        f1 = self.multiple([d, z])
        
        f1 = self.sum(f1, axis=[-1])

        f2 = self.outputD(d)
        # f2 = layers.Reshape((1,))(f2)
        r = self.out([f1, f2])
        r = self.outputActivation(r)
        # print(r.shape)

        return r
    def model(self, inputsize = 64):
        volumeInput = tf.keras.Input(shape=(inputsize, inputsize, inputsize, 1), name='volume')
        parameterInput = tf.keras.Input(shape=(1, 3), name='parameter')

        return tf.keras.models.Model(inputs=[parameterInput, volumeInput], outputs = self.call([parameterInput, volumeInput]))