import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

# IO
sep = seppy.sep()

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
nrec = nrec.astype('int')

nxi = int(2*nvx); dxi = dvx/2; oxi = ovx

wei = geom.coordgeom(nxi,dxi,ny,dy,nz,dz,ox=oxi,nrec=nrec,srcxs=srcx,recxs=recx)

velint = wei.interp_vel(velin,dvx,dy,ovx=ovx)

#datreg = wei.make_sht_cube(dat)
#drx= recx[1] - recx[0]; dsx = srcx[1] - srcx[0]; osx = srcx[0]
#orx = recx[0] - srcx[0]
#sep.write_file("datregbob2.H",datreg.T,ds=[dt,drx,dsx],os=[ot,orx,osx])

img = wei.image_data(dat,dt,ntx=16,minf=1,maxf=51,vel=velint,nhx=20,nrmax=10,nthrds=40)

# Zero-offset image
#sep.write_file("spimgbobfull.H",img,os=[oz,0.0,oxi],ds=[dz,1.0,dxi])

# Subsurface offset
imgt = np.transpose(img,(2,4,3,1,0))  # [nhy,nhx,nz,ny,nx] -> [nz,nx,ny,nhx,nhy]
nhx,ohx,dhx = wei.get_off_axis()
sep.write_file("spimgbobfullext.H",imgt,os=[oz,oxi,0,ohx,0],ds=[dz,dxi,dy,dhx,1.0])

# Angle
#ang = wei.to_angle(img,verb=True,nthrds=40)
#na,oa,da = wei.get_ang_axis()

#sep.write_file("spimgang.H",ang.T,os=[0,oa,ovx],ds=[dz,da,dvx])

