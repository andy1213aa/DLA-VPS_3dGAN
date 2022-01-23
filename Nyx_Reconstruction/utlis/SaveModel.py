import tensorflow as tf
import numpy as np
import datetime
import os
from shutil import copytree, copyfile
class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, gen, dis, dataSetConfig, mode = 'min', save_weights_only = True):
        super(SaveModel, self).__init__()
        self.gen = gen
        self.dis = dis
        # setting directory of saving weight
        self.dataSetConfig = dataSetConfig

        # biggest better or lowest better
        self.mode = mode
        # save type
        self.save_weights_only = save_weights_only
        if mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
        self.counter = 0
        self.training = True
        self.epoch = 1
        self.genDir = dataSetConfig['logDir'] + "gen/"
        self.disDir = dataSetConfig['logDir'] + "dis/"

        # if not os.path.isdir(dataSetConfig['logDir'] + "gen/"):
        #     os.mkdir(dataSetConfig['logDir'] + "gen/")
        # if not os.path.isdir(dataSetConfig['logDir'] + "dis/"):
        #     os.mkdir(dataSetConfig['logDir'] + "dis/")

        # copytree('/work/csun1205/V3/Nyx_Reconstruction', dataSetConfig['logDir']+'Nyx_Reconstruction/')        
        # copyfile('/work/csun1205/V3/main.py', dataSetConfig['logDir'] + 'main.py')
          
    def save_model(self):
        if self.save_weights_only:
            self.gen.save_weights(self.genDir + "trained_ckpt")
            self.dis.save_weights(self.disDir + "trained_ckpt")
        else:
            self.gen.save(self.genDir + "trained_ckpt")
            self.dis.save(self.disDir + "trained_ckpt")
    def save_config(self, monitor_value):
        saveLogTxt = f"""
    Parameter Setting
    =======================================================
    DataSet: { self.dataSetConfig['dataSet']}
    DataShape: ({ self.dataSetConfig['length']}, { self.dataSetConfig['width']}, {self.dataSetConfig['height']})
    DataSize: {self.dataSetConfig['datasize']}
    TrainingSize: { self.dataSetConfig['trainSize']}
    TestingSize: { self.dataSetConfig['testSize']}
    BatchSize: { self.dataSetConfig['batchSize']}
    =======================================================

    Training log
    =======================================================
    Training start: { self.dataSetConfig['startingTime']}
    Training stop: {datetime.datetime.now()}
    Training epoch: {self.epoch}
    Root Mean Square Error: {monitor_value}%
    =======================================================
    """
        with open( self.dataSetConfig['logDir']+'config.txt', 'w') as f:
            f.write(saveLogTxt) 
    def on_epoch_end(self, monitor_value, logs = None):
        # read monitor value from logs
        # monitor_value = logs.get(self.monitor)
        # Create the saving rule
        
        # if self.mode == 'min' and monitor_value < self.best:
            
        #     self.best = monitor_value
        #     self.counter = 0
        # elif self.mode == 'max' and monitor_value > self.best:
            
        #     self.best = monitor_value
        #     self.counter = 0
        # else:
        #     self.counter += 1
        #     if self.counter >= self.dataSetConfig['stopConsecutiveEpoch']:
        #         self.save_model()
        #         self.save_config(monitor_value)
        #         self.training = False
        self.epoch += 1