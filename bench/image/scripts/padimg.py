import numpy as np
import inpout.seppy as seppy
from resfoc.resmig import pad_cft
from resfoc.gain import agc, tpow
from scaas.gradtaper import build_taper, build_taper_ds
from genutils.movie import viewcube3d
import matplotlib.pyplot as plt

sep = seppy.sep()
iaxes,img = sep.read_file("fltimgextprcnew2.H")

[nz,nx,nh] = iaxes.n; [dz,dx,dh] = iaxes.d; [oz,ox,oh] = iaxes.o

img = np.reshape(img,iaxes.n,order='F')
#viewcube3d(tpow(img.T,dz,2.0,transp=True).T,ds=[dz/1000.0,dx/1000.0,dh/1000.0],pclip=0.2,width3=1.0,show=False)
imgt = img.T

nzp = pad_cft(nz); nxp = pad_cft(nx); nhp = pad_cft(nh)

imgp = np.pad(imgt,((0,nhp),(0,nxp),(0,nzp)),'constant')

viewcube3d(tpow(imgp,dz,2.0,transp=True).T,ds=[dz/1000.0,dh/1000.0,dx/1000.0],os=[oz,oh/1000.0,ox],pclip=0.2,width3=1.0,show=False,
           loc1=5.0,loc2=0.0,loc3=2.0)

imgpt = np.zeros(imgp.shape,dtype='float32')

nhrp = imgp.shape[0]; nxrp = imgp.shape[1]; nzrp = imgp.shape[2]
# Vertical taper
tap2l,tap2i = build_taper_ds(nxrp,nzrp,100,250,300,500)

# Midpoint taper
tap2lh,tap2ih = build_taper(nzrp,nxrp,1000,1200)
tap2ihf = np.fliplr(tap2ih.T)

# Offset taper
tap2lo,tap2io = build_taper(nxrp,nhrp,25,55)
tap2iof = np.flipud(tap2io)

print(imgp[:,:,256].shape,tap2io.shape)

plt.figure()
plt.imshow(agc(imgp[:,:,256]),cmap='gray',aspect='auto')
plt.imshow(tap2iof,cmap='jet',alpha=0.3,aspect='auto')
#
#plt.figure()
#plt.imshow(agc(imgp[16]).T,cmap='gray')
#plt.imshow(tap2ihf*tap2i,cmap='jet',alpha=0.3)
#plt.show()

for ih in range(nh):
  imgpt[ih] = tap2ihf.T*tap2i.T*imgp[ih]

for iz in range(nz):
  imgpt[:,:,iz] = imgpt[:,:,iz]*tap2iof

plt.figure()
plt.imshow(imgpt[:,:,256],cmap='gray',aspect='auto')
plt.figure()
plt.imshow(imgp[:,:,256],cmap='gray',aspect='auto')

print(tap2i.shape)

## Create taper
#plt.figure(1)
#plt.imshow(agc(imgt[16]).T,cmap='gray')
#plt.figure(2)
#plt.imshow(agc(imgp[16]).T,cmap='gray')
#plt.imshow(tap2i,cmap='jet',alpha=0.3)
#plt.figure(3)
#plt.imshow(agc(imgp[0]).T,cmap='gray')
#plt.figure(4)
#plt.imshow(agc(imgpt[0]).T,cmap='gray')
#plt.show()

viewcube3d(tpow(imgpt,dz,2.0,transp=True).T,ds=[dz/1000.0,dh/1000.0,dx/1000.0],os=[oz,oh/1000.0,ox],pclip=0.2,width3=1.0,
           loc1=5.0,loc2=0.0,loc3=2.0)
