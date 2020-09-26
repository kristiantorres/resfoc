import inpout.seppy as seppy
from scaas.velocity import insert_circle
import scaas.scaas2dpy as sca2d
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
import numpy as np
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet
from genutils.movie import viewimgframeskey

sep = seppy.sep()

nz,nx = 512,1024
dz,dx = 10,10
vel = np.zeros([nz,nx],dtype='float32') + 3000

velcirc = insert_circle(vel,dz,dx,centerx=5120,centerz=2560,rad=15,val=1500)

nsx = 52; dsx = 20; bx = 100; bz = 100
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=nsx,dsx=dsx,bx=bx,bz=bz)
prp.plot_acq(velcirc,cmap='jet',shpw=True)

ntu = 4500; dtu = 0.001
freq = 20; amp = 100.0; dly = 0.2
wav = ricker(ntu,dtu,freq,amp,dly)

# Model the data
dtd = 0.008
allshot = prp.model_fulldata(velcirc,wav,dtd,dtu,nthrds=26,verb=True)

allshott = np.transpose(allshot,(0,2,1))

sep.write_file("intro_dat.H",allshott.T,ds=[dtd,dx])

#sc = 0.01
#vmin,vmax = sc*np.min(allshot),sc*np.max(allshot)
#plt.imshow(allshot[10],cmap='gray',interpolation='bilinear',vmin=vmin,vmax=vmax); plt.show()
#
## Model the wavefield
#dtw = 0.01; ntw = int(ntu*(dtu/dtw))
#velcircp = prp.pad_model(velcirc)
#[nzp,nxp] = velcircp.shape
#alpha = 0.99
#sca = sca2d.scaas2d(ntw,nxp,nzp,dtw,dx,dz,dtu,bx,bz,alpha)
#
#srcx = prp.allsrcx[26]; srcz = prp.allsrcz[26]
#nsrc = len(srcx)
#
#wfld = np.zeros([ntw,nzp,nxp],dtype='float32')
#sca.fwdprop_wfld(wav,srcx,srcz,nsrc,velcircp,wfld)
#
#viewimgframeskey(wfld,transp=False,pclip=0.1)
#
## Write the wavefield
#sep.write_file("intro_wfld.H",wfld.T,ds=[dx,dz,dtw])

