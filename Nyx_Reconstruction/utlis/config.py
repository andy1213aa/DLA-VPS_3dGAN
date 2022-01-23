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
'datasize' : 2197,
'trainSize' : 800,
'validationSize':8, 
'testSize' : 200,
'batchSize' :32,
'length' : 64,
'width' : 64,
'height' : 64,
'dataType' : 'float',
'numberOfParameter' : 3,
'numberOfParameterDigit' : 7,
'minMax': {'OmM': [0.17, 0.5], 'OmB': [0.03, 0.08], 'h': [0.55, 0.85]},
'stopConsecutiveEpoch' : 100,
'dataSetDir' :  f'/work/csun1205/NyxDataSet/Nyx_tfrecords/NyxDataSet64_64_64_{variable}_1000.tfrecords',
'startingTime' : startingTime,
'logDir' : f'/work/csun1205/NyxDataSet/log/{variable}/Nyx_' + startingDate + '/'
}


dataSet = {'Nyx':Nyx}