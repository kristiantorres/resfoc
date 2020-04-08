import inpout.seppy as seppy
import numpy as np
from utils.movie import viewcube3d

sep = seppy.sep([])

#Inline 112
daxes,dat = sep.read_file('./dat/invfloat.H',form='native')

dat = dat.reshape(daxes.n,order='F')

datw = dat[20:276,160:,40:168]
ds = [0.01,0.02,0.02]
os = [20*0.01,160*0.02,50*0.02]

viewcube3d(datw,ds=ds,os=os,
    pclip=0.95,interp='sinc',width1=8,label1='X (km)',label2='X (km)',label3='Z (km)',
    transp=False,cbar=False,cmap='gray')

#axes = seppy.axes(datw.shape,os,ds)
#sep.write_file(None,axes,datw,ofname='elf.H')

#datwt = np.transpose(datw,(0,2,1))
#ds = [0.01,0.02,0.02]
#os = [20*0.01,50*0.02,160*0.02]
#
#viewcube3d(datwt,ds=[0.01,0.02,0.02],os=[20*0.01,50*0.02,160*0.02],
#    pclip=0.95,interp='sinc',width3=8,label1='X (km)',label2='X (km)',label3='Z (km)',
#    transp=False,cbar=False,cmap='gray')

#axes = seppy.axes(datwt.shape,os,ds)
#sep.write_file(None,axes,datwt,ofname='elft.H')
