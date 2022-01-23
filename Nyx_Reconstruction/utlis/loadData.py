from ..utlis import config
import numpy as np 
import os
import tensorflow as tf
from shutil import copyfile
def generateData(dataSetConfig):
    
    dataType = {'float': [4, tf.float32]}
    
    def _parse_function(example_proto):
        features = tf.io.parse_single_example(
            example_proto,
            features={
                "Parameter1": tf.io.FixedLenFeature([], tf.float32),
                "Parameter2": tf.io.FixedLenFeature([], tf.float32),
                "Parameter3": tf.io.FixedLenFeature([], tf.float32),
                'data_raw': tf.io.FixedLenFeature([], tf.string)
            }
        )
     
        P1 = features['Parameter1']
        P2 = features['Parameter2']
        P3 = features['Parameter3']
        data = features['data_raw']
     
        data = tf.io.decode_raw(data, tf.float32)
        
        if dataSetConfig['variable'] == 'density':
     
            maxi = 492655700000.0
            mini = 562485060.0
      

        elif dataSetConfig['variable'] == 'xmom':
 
            maxi = 62882016000000.0
            mini = -70025482000000.0

        elif dataSetConfig['variable'] == 'ymom':
   
            maxi = 180579870000000.0
            mini = -124431645000000.0
        

        elif dataSetConfig['variable'] == 'Temp':
        
            maxi = 2361152.0
            mini = 2.4191138e-09

        elif dataSetConfig['variable'] == 'rho_e':
            maxi = 2.3533318e+16
            mini = 0.42123172
      
        elif dataSetConfig['variable'] == 'zmom':
            maxi = 63409750000000.0
            mini = -80423330000000.0

        elif dataSetConfig['variable'] == 'particle_mass_density':
            maxi = 1657542300000.0
            mini = 0.0


        data = (data - mini) / (maxi-mini)
        data = data*2 - 1 #rescale the value range to [-1, 1].
     
        data = tf.reshape(data, [dataSetConfig['height'], dataSetConfig['width'], dataSetConfig['length'], 1])
        P1 = tf.reshape(P1, [1])
        P2 = tf.reshape(P2, [1])
        P3 = tf.reshape(P3, [1])
        P = tf.stack([P1, P2, P3], axis=1)
        return P, data
        
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = tf.data.TFRecordDataset(dataSetConfig['dataSetDir'])
    data = data.map(_parse_function, num_parallel_calls=AUTOTUNE)


    data = data.shuffle(dataSetConfig['trainSize'], reshuffle_each_iteration=True)


    data_batch = data.batch(dataSetConfig['batchSize'], drop_remainder = True)

    data_batch = data_batch.prefetch(buffer_size = AUTOTUNE)
    return data_batch