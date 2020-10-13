import inpout.seppy as seppy
import numpy as np
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Read in the original data
daxes,dat = sep.read_file("hale_shotflatbob.H")
dat = dat.reshape(daxes.n,order='F')
[nt,ntr] = daxes.n; [ot,otr] = daxes.o; [dt,dtr] = daxes.d

# Read in the original image
iaxes,img = sep.read_file("spimgbobdistr.H")
img = img.reshape(iaxes.n,order='F')
[nz,ny,nx] = iaxes.n; [oz,oy,ox] = iaxes.o; [dz,oy,dx] = iaxes.d

# Read in the denoised data
daxes,den = sep.read_file("hale_shotflatbobden.H")
den = den.reshape(daxes.n,order='F')

# Read in the denoised image
iaxes,dmg = sep.read_file("spimgbobdistrden.H")
dmg = dmg.reshape(iaxes.n,order='F')

datcmp = np.zeros([2,nt,ntr],dtype='float32')
datcmp[0] = dat
datcmp[1] = den

viewimgframeskey(datcmp[:,:,300:600],transp=False,zmin=0,zmax=nt*dt,pclip=0.3,interp='bilinear',show=False)

imgcmp = np.zeros([2,nz,ny,nx],dtype='float32')

imgcmp[0] = img
imgcmp[1] = dmg

viewimgframeskey(imgcmp[:,:,0,:],transp=False,zmin=0,zmax=nz*dz,
                 xmin=ox,xmax=ox+nx*dx,interp='bilinear',pclip=0.3)

