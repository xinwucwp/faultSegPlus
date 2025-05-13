#from numpy.random import seed
#seed(12345)
#from tensorflow import set_random_seed
#set_random_seed(1234)
import tensorflow as tf
import os
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
from utilsarguvsig import DataGenerator
from unetsig import *
from loss_functions0 import *

#from fnet import *
fpath = '/home/yli/fseg/'

s = Semantic_loss_functions()
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':1,
          'dim':(128,128,128),
          'n_channels':1,
          'shuffle': True}
  #seismPathT = "../data/train/seis/"
  #faultPathT = "../data/train/fault/"
  seismPath = fpath+"data/nx900/"
  faultPath = fpath+"data/fx900/"
  train_ID=[]
  valid_ID=[]
  for sfile in os.listdir(seismPath):
    if sfile.endswith(".dat"):
       if(int(sfile[:-4])<700):
         train_ID.append(sfile)
       elif(int(sfile[:-4])>=700 and int(sfile[:-4])<800):
          valid_ID.append(sfile)
  train_generator = DataGenerator(dpath=seismPath,fpath=faultPath,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPath,fpath=faultPath,
                                  data_IDs=valid_ID,**params)
  model = unet(input_size=(None, None, None,1))
  #model = resnet(input_size=(None, None, None,1))
  model.compile(optimizer=Adam(lr=1e-4), loss=s.cebdice,metrics = ['accuracy',s.f1_socre,s.dice,s.sensitivity,s.precision,s.specificity])
  model.summary()

  # checkpoint
  filepath=fpath+"3-4new/fseg-{epoch:02d}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
        verbose=1, save_best_only=False, mode='max')
  reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, 
                                patience=2, min_lr=1e-8)
  callbacks_list = [checkpoint, reduce_lr]
  print("data prepared, ready to train!")
  # Fit the model
  history=model.fit_generator(generator=train_generator,
  validation_data=valid_generator,epochs=130,callbacks=callbacks_list,verbose=1)
  model.save(fpath+'3-4new/fseg.hdf5')
  showHistory(history)

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  fig = plt.figure(figsize=(10,6))

  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()
  plt.savefig('history for accuracy 3-4new.png')
  plt.close()

  # summarize history for loss
  fig = plt.figure(figsize=(10,6))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()
  plt.savefig('history for loss 3-4new.png')
  plt.close()

  plt.plot(history.history['f1_socre'])
  plt.plot(history.history['val_f1_socre'])
  plt.title('Model f1_socre',fontsize=20)
  plt.ylabel('Accuracy',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'test'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()
  plt.savefig('history for f1_socre 3-4new.png')
  plt.close()

  with open ('3-4new-metrics_loss.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['loss']))
  with open ('3-4new-metrics_val_loss.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_loss']))

  with open ('3-4new-metrics_f1_socre.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['f1_socre']))
  with open ('3-4new-metrics_val_f1_socre.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_f1_socre']))

  with open ('3-4new-metrics_acc.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['acc']))
  with open ('3-4new-metrics_val_acc.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_acc']))

  with open ('3-4new-metrics_dice.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['dice']))
  with open ('3-4new-metrics_val_dice.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_dice']))
  
  with open ('3-4new-metrics_sensitivity.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['sensitivity']))
  with open ('3-4new-metrics_val_sensitivity.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_sensitivity']))
  
  with open ('3-4new-metrics_precision.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['precision']))
  with open ('3-4new-metrics_val_precision.txt','a',encoding='utf-8') as f:
    f.write(str(history.history['val_precision']))


if __name__ == '__main__':
    main()
