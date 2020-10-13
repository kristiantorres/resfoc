import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
import matplotlib.pyplot as plt

# IO
sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("hale_shotflatsort.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in the velocity model
vaxes,vel = sep.read_file("vintz.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("hale_srcxflat.H")
raxes,recx = sep.read_file("hale_recxflatsort.H")
_,nrec= sep.read_file("hale_nrec.H")
nrec = nrec.astype('int')

nxi = int(2*nvx); dxi = dvx/2; oxi = ovx

wei = geom.coordgeom(nxi,dxi,ny,dy,nz,dz,ox=oxi,nrec=nrec,srcxs=srcx,recxs=recx)

velint = wei.interp_vel(velin,dvx,dy,ovx=ovx)

plt.imshow(velint[:,0,:],cmap='jet')
plt.show()

img = wei.image_data(dat,dt,ntx=16,minf=1,maxf=26,vel=velint,nhx=0,nrmax=10,nthrds=40)

# Zero-offset image
sep.write_file("spimg.H",img,os=[oz,0.0,oxi],ds=[dz,1.0,dxi])

# Subsurface offset
#imgt = np.transpose(img,(2,4,3,1,0))  # [nhy,nhx,nz,ny,nx] -> [nz,nx,ny,nhx,nhy]
#nhx,ohx,dhx = wei.get_off_axis()
#sep.write_file("spimgext2.H",imgt,os=[oz,oxi,0,ohx,0],ds=[dz,dxi,dy,dhx,1.0])

# Angle
#ang = wei.to_angle(img,verb=True,nthrds=40)
#na,oa,da = wei.get_ang_axis()

#sep.write_file("spimgang.H",ang.T,os=[0,oa,ovx],ds=[dz,da,dvx])

