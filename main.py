
import tensorflow as tf

from Nyx_Reconstruction.utlis.loadData import generateData
from Nyx_Reconstruction.utlis import config
from Nyx_Reconstruction.utlis.SaveModel import SaveModel
from Nyx_Reconstruction.model.generator import generator
from Nyx_Reconstruction.model.discriminator import discriminator
from Nyx_Reconstruction.utlis.loss_function import generator_loss, discriminator_loss
import numpy as np

def main():


    @tf.function
    def train_generator(real_data):
        with tf.GradientTape() as tape:
            fake_data_by_random_parameter = gen(real_data[0] ,training = True)  #generate by random parameter

            gFake_logit = dis([real_data[0], fake_data_by_random_parameter],training = True)
            gFake_loss = generator_loss(gFake_logit)
            disparate = 5e-1*tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.math.abs(fake_data_by_random_parameter-real_data[1])**2, axis=[1, 2, 3, 4])))
            gLoss = gFake_loss + disparate
        gradients = tape.gradient(gLoss, gen.trainable_variables)
        genOptimizer.apply_gradients(zip(gradients, gen.trainable_variables))
        return gFake_loss, gFake_logit, disparate
    
    @tf.function
    def train_discriminator(real_data):
        print('DIS GRAPH SIDE EFFECT')
        with tf.GradientTape() as t:

            fake_data = gen(real_data[0],training = True)
            real_logit = dis([real_data[0], real_data[1]] ,training = True)
            fake_logit = dis([real_data[0], fake_data],training = True)
            real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)
            dLoss = (fake_loss + real_loss)

        D_grad = t.gradient(dLoss, dis.trainable_variables)
        disOptimizer.apply_gradients(zip(D_grad, dis.trainable_variables))
        return real_loss ,  fake_loss,  real_logit,fake_logit
    
    
    
    dataSetConfig = config.dataSet['Nyx'] #What data you want to load
  
    gen = generator().model()
    gen.summary()
    dis = discriminator().model(64)
    dis.summary()
    
    
    disOptimizer = tf.keras.optimizers.RMSprop(lr = 2e-4, decay = 1e-4)
    genOptimizer = tf.keras.optimizers.RMSprop(lr = 5e-5,decay = 1e-4)
                
    training_batch = generateData(dataSetConfig)
    
    summary_writer = tf.summary.create_file_writer(dataSetConfig['logDir'])
    # No use 'min' to decide end the training or not
    # Instead directly use epoch
    saveModel = SaveModel(gen, dis, dataSetConfig, mode = 'min', save_weights_only=dataSetConfig['save_weights_only'])   #Build a training rule
  

    while saveModel.training and saveModel.epoch < dataSetConfig['epochs']:  
       # Average_percentage = 0
        for step, real_data in enumerate(training_batch):
            # real_data 中 real_data[0] represents three input parameters, i.e. real_data[0][0] real_data[0][1] and real_data[0][2]. 
            # real_data[1] represents raw data.
        
            dRealLoss, dFakeLoss, dReal_logit,dFake_logit= train_discriminator(real_data)
            with summary_writer.as_default():
                tf.summary.scalar('discriminator_loss_D(x)', dRealLoss, disOptimizer.iterations)
                tf.summary.scalar('discriminator_loss_D(G(z))', dFakeLoss, disOptimizer.iterations)
                tf.summary.scalar('discriminator_loss_Total', dRealLoss+dFakeLoss, disOptimizer.iterations)

            gLoss, glogit, gDisparate= train_generator(real_data)
            with summary_writer.as_default():
                tf.summary.scalar('generator_loss_D(g(z))', gLoss, genOptimizer.iterations)
                tf.summary.scalar('generator_disparate', gDisparate, genOptimizer.iterations)
                tf.summary.scalar('generator_loss_Total', gDisparate+gLoss, genOptimizer.iterations)
         
        print(f'Epoch: {saveModel.epoch:6} Batch: {step:3} Disparate:{gDisparate:4.5} G_loss: {gLoss:4.5} D_real_loss: {dRealLoss:4.5} D_fake_loss: {dFakeLoss:4.5}')        
            

        saveModel.on_epoch_end()
        # Save the model for given epoch.
        if saveModel.epoch%dataSetConfig['save_epochs'] == 0:
            saveModel.save_model()
            
if __name__ == "__main__":
    main()