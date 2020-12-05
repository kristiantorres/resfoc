import inpout.seppy as seppy
import numpy as np
import oway.streamergeom as geom
from oway.utils import interp_vel
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey
from genutils.plot import plot_vel2d, plot_img2d

# IO
sep = seppy.sep()

# Read in geometry

# Read in data
daxes,dat = sep.read_file("hale_shotflatbob.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in the velocity model
vaxes,vel = sep.read_file("vintzcomb.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int32')

nxi = int(2*nvx); dxi = dvx/2; oxi = ovx

wei = geom.streamergeom(nxi,dxi,ny,dy,nz,dz,ox=oxi,nrec=nrec,srcxs=srcx,recxs=recx,maxx=350*dxi)

velint = interp_vel(nz,ny,0.0,1.0,nxi,oxi,dxi,velin,dvx,1.0,ovx)
#plot_vel2d(velint[:,0,:])
velint[:,:,:] = 2.0

img = wei.image_data(dat,dt,ntx=16,minf=1,maxf=51,vel=velint,nhx=0,nrmax=10,nthrds=40,sverb=True)

plot_img2d(img[:,0,:],pclip=0.4)

# Zero-offset image
#sep.write_file("spimgboborigtest.H",img,os=[oz,0.0,oxi],ds=[dz,1.0,dxi])


