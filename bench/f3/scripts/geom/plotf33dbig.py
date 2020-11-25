import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from genutils.movie import viewcube3d
from genutils.plot import plot3d

sep = seppy.sep([])

#Inline 112
daxes,dat = sep.read_file('./mig/mig.H')

dat = dat.reshape(daxes.n,order='F').astype('float32')

datw = dat[:,200:1200,5:505]

ds = [0.004,0.025,0.025]
os = [300*ds[0],0.0,0.0]

#nans = np.isnan(dat)
#dat[nans] = 0

#datsm = smooth(dat,rect1=100,rect2=100,rect3=100)

viewcube3d(datw,ds=ds,os=os,pclip=0.2,loc1=6.5,loc2=7.2,loc3=2.0,
          interp='none',width1=8,label1='X (km)',label2='Y (km)',label3='Time (s)',
          transp=False,cbar=False,cmap='gray',labelsize=15,ticksize=15)

#plot3d(datw[300:700,:,:],ds=ds,os=os,pclip=0.2,loc1=6.5,loc2=7.2,loc3=2.0,
#       interp='none',width1=8,label1='X (km)',label2='Y (km)',label3='Time (s)',
#       transp=False,cbar=False,cmap='gray',labelsize=15,ticksize=15,
#       figname='./fig/img3d.png')
#
