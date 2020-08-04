import inpout.seppy as seppy
import numpy as np
import oway.coordgeomnode as geom
import matplotlib.pyplot as plt
from dask.distributed import SSHCluster, LocalCluster, Client

sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("sigsbee_shotflat.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

nr = 348; nw = 100
datw = dat[0:nr*nw,:]

# Read in velocity model
vaxes,vel = sep.read_file("sigsbee_vel.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("sigsbee_srcxflat.H")
raxes,recx = sep.read_file("sigsbee_recxflat.H")
_,nrec = sep.read_file("sigsbee_nrec.H")
nrec = nrec.astype('int')

srcxw = srcx[0:nw]
recxw = recx[0:nr*nw]
nrecw = nrec[0:nw]

cluster = SSHCluster(
                     ["localhost", "fantastic", "thing", "torch", "jarvis"],
                     connect_options={"known_hosts": None},
                     worker_options={"nthreads": 1, "nprocs": 1, "memory_limit": 20e9},
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}
                    )

client = Client(cluster)

# Imaging grid (same as shots)
nxi = saxes.n[0]; oxi = srcx[0]; dxi = srcx[1] - srcx[0]
print("Image grid: nxi=%d oxi=%f dxi=%f"%(nxi,oxi,dxi))

wei = geom.coordgeomnode(nxi,dxi,ny,dy,nz,dz,ox=oxi,nrec=nrecw,srcx=srcxw,recx=recxw)

velint = wei.interp_vel(velin,dvx,dy,ovx=ovx)

img = wei.image_data(datw,dt,ntx=16,minf=1,maxf=51,vel=velint,nhx=0,nrmax=20,
                     nthrds=40,client=client)

sep.write_file("mysigimg.H",img,ds=[dz,dy,dxi],os=[oz,0.0,oxi])

