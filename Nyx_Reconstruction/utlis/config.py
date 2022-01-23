import datetime

# logfile dirs name by time.
#============================
startingTime=datetime.datetime.now()
startingDate = f'{startingTime.year}_{startingTime.month}_{startingTime.day}'+'_'+f'{startingTime.hour}_{startingTime.minute}'
variable = 'rho_e'
#============================


Nyx={
'dataSet' : 'Nyx',
'variable' : variable,
'epochs' : 500,
'datasize' : 1000,
'trainSize' : 800,
'testSize' : 200,
'batchSize' :32,
'length' : 64,
'width' : 64,
'height' : 64,
'minMax': {'OmM': [0.17, 0.5], 'OmB': [0.03, 0.08], 'h': [0.55, 0.85]},
'stopConsecutiveEpoch' : 100,
'dataSetDir' :  f'C:/Users/andy1/Desktop/DLA-VPS/Nyx_tfrecords/NyxDataSet64_64_64_{variable}_1000.tfrecords',
'startingTime' : startingTime,
'logDir' : f'C:/Users/andy1/Desktop/DLA-VPS/log/{variable}/Nyx_' + startingDate + '/',
'save_weights_only': False,
'save_epochs': 10,
}

# add more data set in future.
dataSet = {'Nyx':Nyx}