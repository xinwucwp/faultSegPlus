import numpy as np
import keras
import random
from keras.utils import to_categorical
from scipy import interpolate

class DataGenerator2(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,dpath,fpath,data_IDs, batch_size=1, dim=(256,128,128), 
             n_channels=1, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    A,Y = self.__data_generation(data_IDs_temp)

    return A,Y


  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    id = str(data_IDs_temp[0])
    id = int(id[:-4])
    a = 2 #data augumentation
    if (id % 2) == 0 and id < 800:
      A = np.zeros((a*self.batch_size,256,128,128, self.n_channels),dtype=np.single)
      Y = np.zeros((a*self.batch_size,256,128,128,1),dtype=np.single)
    else:
      A = np.zeros((a*self.batch_size,128,128,128, self.n_channels),dtype=np.single)
      Y = np.zeros((a*self.batch_size,128,128,128,1),dtype=np.single)
    

    for k in range(self.batch_size):
      gx  = np.fromfile(self.dpath+str(data_IDs_temp[k]),dtype=np.single)
      fx  = np.fromfile(self.fpath+data_IDs_temp[k],dtype=np.single)
      if (id % 2) == 0 and id < 800:
        gx = np.reshape(gx,(128,128,256))
        fx = np.reshape(fx,(128,128,256))
        gx = datanormalize(gx)
        fx = np.clip(fx,0,1)
        gx = np.transpose(gx)
        fx = np.transpose(fx)
      else:
        gx = np.reshape(gx,(128,128,128))
        fx = np.reshape(fx,(128,128,128))
        gx = datanormalize(gx)
        fx = np.clip(fx,0,1)
        gx = np.transpose(gx)
        fx = np.transpose(fx)
      #in seismic processing, the dimensions of a seismic array is often arranged as
      #a[n3][n2][n1] where n1 represnts the vertical dimenstion. This is why we need 
      #to transpose the array here in python 
      c = k*a
      A[c+0,:,:,:,0] = gx
      Y[c+0,:,:,:,0] = fx
      r = random.randint(0,3)
      gx = np.rot90(gx,r,(2,1))
      fx = np.rot90(fx,r,(2,1))
      A[c+1,:,:,:,0] = gx
      Y[c+1,:,:,:,0] = fx
    return A,Y

def datanormalize(x):
  gm = np.mean(x)
  gs = np.std(x)
  g = x-gm
  g = g/gs
  return g

'''def dataprocess(x):
  gm = np.mean(x)
  gs = np.std(x)
  g = x-gm
  g = g/gs
  g = np.transpose(g)
  return g

def dataupsample(x):
  #x = np.transpose(x)
  x1 = np.linspace(0,128,128)
  #print(x1[0:10])
  x_new = np.linspace(0,128,256)
  gxu = np.zeros((256,128,128),dtype="float32")
  for i in range(128):
      for j in range(128):
              tck = interpolate.splrep(x1,x[:,i,j])
              gxu[:,i,j] = interpolate.splev(x_new,tck)
  return gxu

def faultupsample(x):
  #x = np.transpose(x)
  x1 = np.linspace(0,128,128)
  #print(x1[0:10])
  x_new = np.linspace(0,128,256)
  gxu = np.zeros((256,128,128),dtype="float32")
  for i2 in range(128):
      for i3 in range(128):
              tck = interpolate.splrep(x1,x[:,i2,i3])
              gxu[:,i2,i3] = interpolate.splev(x_new,tck)
              for i1 in range(256):
                if gxu[i1,i2,i3]<0.5:
                  gxu[i1,i2,i3]=0
                else:
                  gxu[i1,i2,i3]=2
  return gxu'''
