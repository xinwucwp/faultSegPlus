import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
#from unetd import mybce
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
path = '/media/xinwu/disk-2/cig-sl/'
datapath = '/media/xinwu/disk-1/oz/'

def main(argv):
  loadModel(argv[1])
  #goXin()
  #goD2()
  #goNz()
  #goDy()
  #goPttep()
  #goCrf()
  #goZ3()
  goBgp()
  #goCnpc()
  #goResample()
  #goResampleX()
  #goOz()

def loadModel(mk):
  global model
  model = load_model(path+'check3/fseg-'+mk+'.hdf5')

def goOz(): 
  fname = "ggr.dat"
  fpath = datapath+"data/"
  n1,n2,n3=626,401,501
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fgr.dat",format="%4")

def goCnpc(): 
  fname = "sr.dat"
  fpath = datapath+"data/seis/"
  n1,n2,n3=1500,968,611
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2=400,480
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fp.dat",format="%4")
  os.popen('./goDisplay cnpc').read()


def goZ3(): 
  fname = "gx.dat"
  fpath = path+"data/prediction/z3/"
  n1,n2,n3=251,901,1592
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m3 = 816
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fp.dat",format="%4")
  os.popen('./goDisplay z3').read()

def goXin(): 
  fname = "gr.dat"
  fpath = path+"data/prediction/xin/"
  n1,n2,n3=1856,231,311
  n1,n2,n3=1400,231,311
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fpr.dat",format="%4")
  os.popen('./goDisplay xin').read()

def goD2(): 
  fname = "gx.dat"
  fpath = path+"data/prediction/d2/"
  n1,n2,n3=251,771,991
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fpx.dat",format="%4")
  os.popen('./goDisplay d2').read()

def goNz(): 
  fname = "gr.dat"
  fpath = path+"data/prediction/nz/"
  n1,n2,n3=750,441,601
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fpr.dat",format="%4")
  os.popen('./goDisplay nz').read()

def goDy(): 
  fname = "rx.dat"
  fpath = path+"data/prediction/dy/"
  n1,n2,n3=1251,701,801
  #n1,n2,n3=626,350,400
  m1 = int(np.ceil(n1/16)*16)
  m1 = 320
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fpx.dat",format="%4")
  os.popen('./goDisplay dy').read()

def goBgp(): 
  fname = "gx.dat"
  fpath = path+"data/prediction/bgp/"
  n1,n2,n3=1250,240,140
  n1,n2,n3=1250,481,281
  n1,n2,n3=2501,481,281
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1 = 1264
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fp.dat",format="%4")
  #os.popen('./goDisplay bgp').read()

def goPttep(): 
  fname = "sx.dat"
  fpath = path+"data/prediction/pttep/"
  n1,n2,n3=1250,240,140
  n1,n2,n3=750,1193,863
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=400,480,864
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fpx.dat",format="%4")
  os.popen('./goDisplay pttep').read()
def goCrf(): 
  fname = "gxl.dat"
  fpath = path+"data/prediction/crf/"
  n1,n2,n3=601,3675,825
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m1,m2=400,480
  m3 = int(np.ceil(n3/16)*16)
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  fx.tofile(fpath+"fpx-3-80.dat",format="%4")
  os.popen('./goDisplay crf').read()

def goResample():
  fname = "gx.dat"
  fpath = path+"data/prediction/bgp/"
  n1,n2,n3 = 251,771,991
  n1,n2,n3=1251,701,801
  n1,n2,n3=2501,481,281
  #n1,n2,n3=1501,441,601
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  n1,n2,n3=gx.shape 
  m1 = int(np.round(n1/2))
  print(m1)
  gr = np.zeros((m1,n2,n3),dtype=np.single)
  for i1 in range(m1):
    for i2 in range(n2):
      for i3 in range(n3):
        gr[i1,i2,i3] = gx[i1*2,i2,i3]
  gr = np.transpose(gr)
  gr.tofile(fpath+"gr.dat",format="%4")

def goResampleX():
  fname = "gx.dat"
  fpath = path+"data/prediction/bgp/"
  n1,n2,n3=1251,701,801
  n1,n2,n3=2501,481,281
  gx = loadData(n1,n2,n3,fpath,fname) #load seismic
  n1,n2,n3=gx.shape 
  m1 = int(np.round(n1/2))
  m2 = int(np.round(n2/2))
  m3 = int(np.round(n3/2))
  print(m1)
  print(m2)
  print(m3)
  gr = np.zeros((m1,m2,m3),dtype=np.single)
  for i1 in range(m1):
    for i2 in range(m2):
      for i3 in range(m3):
        gr[i1,i2,i3] = gx[i1*2,i2*2,i3*2]
  gr = np.transpose(gr)
  gr.tofile(fpath+"gr2.dat",format="%4")

def goPredict(m1,m2,m3,gx):
  n1,n2,n3=gx.shape 
  if m1>=n1 and m2>=n2 and m3>=n3:
    return goPredictFull(m1,m2,m3,gx)
  else:
    return goPredictSubs(m1,m2,m3,gx)

def goPredictFull(m1,m2,m3,gx): 
  n1,n2,n3=gx.shape 
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  gk = np.zeros((1,m1,m2,m3,1),dtype=np.single)
  gk[0,:n1,:n2,:n3,0] = gx[:,:,:]
  fk = model.predict(gk,verbose=1) #fault prediction
  fx[:,:,:] = fk[0,:n1,:n2,:n3,0]
  #set the bounds
  fx[-1,:,:]=fx[-2,:,:]
  fx[:,-1,:]=fx[:,-2,:]
  fx[:,:,-1]=fx[:,:,-2]
  return fx

#m1,m2,m3:the dimensions of a subset
#each needs be divisible by 16,
#choose large dimensions if your CPU/GPU memory allows
def goPredictSubs(m1,m2,m3,gx): 
  n1,n2,n3=gx.shape 
  p1,p2,p3=16,16,16 #overlap
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  c1=1+int(np.ceil(float(n1-m1)/(m1-p1)))
  c2=1+int(np.ceil(float(n2-m2)/(m2-p2)))
  c3=1+int(np.ceil(float(n3-m3)/(m3-p3)))
  for k3 in range(c3):
    for k2 in range(c2):
      for k1 in range(c1):
        gk = np.zeros((m1,m2,m3),dtype=np.single)
        b1,b2,b3 = k1*(m1-p1),k2*(m2-p2),k3*(m3-p3)
        e1,e2,e3 = b1+m1,b2+m2,b3+m3
        e1 = min(e1,n1)
        e2 = min(e2,n2)
        e3 = min(e3,n3)
        d1,d2,d3 = e1-b1,e2-b2,e3-b3
        gk[0:d1,0:d2,0:d3] = gx[b1:e1,b2:e2,b3:e3]
        gm,gs = np.mean(gk),np.std(gk)
        gk = gk-gm
        gk = gk/gs
        gk = np.reshape(gk,(1,m1,m2,m3,1))
        fk = model.predict(gk,verbose=1) #fault prediction
        t1 = min(int(p1/2),b1)
        t2 = min(int(p2/2),b2)
        t3 = min(int(p3/2),b3)
        fx[b1+t1:e1,b2+t2:e2,b3+t3:e3] = fk[0,t1:d1,t2:d2,t3:d3,0]
  #set the bounds
  fx[-1,:,:]=fx[-2,:,:]
  fx[:,-1,:]=fx[:,-2,:]
  fx[:,:,-1]=fx[:,:,-2]
  return fx


def loadData(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  #gmin,gmax=np.min(gx)/5,np.max(gx)/5
  gm,gs = np.mean(gx),np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

def loadDatax(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

def sigmoid(x):
    s=1.0/(1.0+np.exp(-x))
    return s

def plot2d(gx,fx,fp,at=1,png=None):
  fig = plt.figure(figsize=(15,5))
  #fig = plt.figure()
  ax = fig.add_subplot(131)
  ax.imshow(gx,vmin=-2,vmax=2,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(132)
  ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(133)
  ax.imshow(fp,vmin=0,vmax=1.0,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  if png:
    plt.savefig(pngDir+png+'.png')
  #cbar = plt.colorbar()
  #cbar.set_label('Fault probability')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
    main(sys.argv)


