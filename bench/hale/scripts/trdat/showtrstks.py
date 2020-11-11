import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewimgframeskey
from genutils.plot import plot_img2d, plot_allanggats

sep = seppy.sep()

fw = 0; nw = 50
# Read in focused images
faxes,foc = sep.read_wind("hale_foctrimgs.H",fw=fw,nw=nw)
[nz,na,ny,nx,nmf] = faxes.n; [oz,oa,oy,ox,om] = faxes.o; [dz,da,dy,dx,dm] = faxes.d
foc = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
focw = foc[:,:,0,31:,:]
focww = focw[:,140:652,:,100:356]
focstk = np.sum(focw,axis=2)
gfoc = agc(focstk)[:,140:652,100:356]
#plot_allanggats(focww[0],dz=dz,dx=dx,jx=5,aagc=False,interp='bilinear',aspect=2.0)

# Read in defocused images
daxes,dfc = sep.read_wind("hale_deftrimgs.H",fw=fw,nw=nw)
[nz,na,ny,nx,nmd] = daxes.n
dfc = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
dfcw = dfc[:,:,0,31:,:]
dfcstk = np.sum(dfcw,axis=2)
gdfc = agc(dfcstk)[:,140:652,100:356]

# Read in residually defocused images
raxes,res = sep.read_wind("hale_restrimgs.H",fw=fw,nw=nw)
res = np.ascontiguousarray(res.reshape(raxes.n,order='F').T).astype('float32')
resw = res[:,:,0,31:,:]
resstk = np.sum(res[:,:,0,:,:],axis=2)
gres = agc(resstk)[:,140:652,100:356]

nxw,nzw = gres.shape[-2:]

tot = np.zeros([2*nmf+nmd,nxw,nzw],dtype='float32')

tot[0::3,:,:] = gfoc[:]
tot[1::3,:,:] = gdfc[:]
tot[2::3,:,:] = gres[:]

viewimgframeskey(tot,pclip=0.5,interpolation='bilinear',show=True,dz=dz,dx=dx)

