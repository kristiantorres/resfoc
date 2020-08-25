import inpout.seppy as seppy
import numpy as np
import oway.coordgeomnode as geom
from oway.coordgeomnode import create_outer_chunks
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey
from dask.distributed import SSHCluster, Client

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

nochnks = 4
ochunks = create_outer_chunks(nochnks,dat,nrec,srcx=srcx,recx=recx)

nxi = int(2*nvx); dxi = dvx/2; oxi = ovx

# Create dask cluster
cluster = SSHCluster(
                     ["localhost", "fantastic", "thing", "jarvis", "storm"],
                     connect_options={"known_hosts": None},
                     worker_options={"nthreads": 1, "nprocs": 1, "memory_limit": 20e9},
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}
                    )

client = Client(cluster)

img = np.zeros([nz,ny,nxi],dtype='float32')
for k in range(nochnks):
  # Get data for outer chunk
  datw  = ochunks[k]['dat' ]; nrecw = ochunks[k]['nrec']
  srcxw = ochunks[k]['srcx']; recxw = ochunks[k]['recx']
  # Build geometry object
  wei = geom.coordgeomnode(nxi,dxi,ny,dy,nz,dz,ox=oxi,nrec=nrecw,srcx=srcxw,recx=recxw)
  if(k == 0):
    velint = wei.interp_vel(velin,dvx,dy,ovx=ovx)
  # Distributed imaging
  img += wei.image_data(datw,dt,ntx=16,minf=1,maxf=51,vel=velint,nrmax=10,
                        nthrds=40,client=client)

# Zero-offset image
sep.write_file("spimgbob.H",img,os=[oz,0.0,oxi],ds=[dz,1.0,dxi])
