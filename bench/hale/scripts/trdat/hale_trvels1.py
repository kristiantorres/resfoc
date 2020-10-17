import inpout.seppy as seppy
import numpy as np
from velocity.stdmodels import velfaultsrandom
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the image
iaxes,img = sep.read_file("spimgbobdistr.H")
img = img.reshape(iaxes.n,order='F').T
sc = 0.4
imin = np.min(img)*sc; imax = np.max(img)*sc

nx = 1000; nz = 900
vel,ref,cnv,lbl = velfaultsrandom(nx=nx,nz=nz,layer=70,maxvel=3000)
dz = 0.005; dx = 0.01675; ox = 7.035

sep.write_file("hale_veltr2.H",vel,os=[0,ox],ds=[dz,dx])
sep.write_file("hale_reftr2.H",ref,os=[0,ox],ds=[dz,dx])
sep.write_file("hale_cnvtr2.H",cnv,os=[0,ox],ds=[dz,dx])
sep.write_file("hale_lbltr2.H",lbl,os=[0,ox],ds=[dz,dx])

#plt.figure()
#plt.imshow(vel,cmap='jet',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0])
#plt.imshow(vel[:,200:-200],cmap='jet',interpolation='bilinear')
#plt.figure()
#plt.imshow(img[:,0,:].T,cmap='gray',interpolation='bilinear',extent=[ox,ox+nx*dx,nz*dz,0],vmin=imin,vmax=imax)
#plt.imshow(img[:,0,:].T,cmap='gray',interpolation='bilinear',vmin=imin,vmax=imax)
#plt.show()

