import inpout.seppy as seppy
import numpy as np
import oway.coordgeomnode as geom
from oway.coordgeomnode import create_outer_chunks
import matplotlib.pyplot as plt
from dask.distributed import SSHCluster, LocalCluster, Client
from cluster.daskutils import shutdown_sshcluster

sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("sigsbee_shotflat.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in velocity model
vaxes,vel = sep.read_file("sigsbee_veloverw2.H")
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

nochnks = 5
ochunks = create_outer_chunks(nochnks,dat,nrec,srcx=srcx,recx=recx)

hosts = ["localhost", "fantastic", "thing", "storm", "jarvis"]
cluster = SSHCluster(
                     hosts,
                     connect_options={"known_hosts": None},
                     worker_options={"nthreads": 1, "nprocs": 1, "memory_limit": 20e9,
                                     "worker_port": '33149:33150'},
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}
                    )

client = Client(cluster)

# Imaging grid (same as shots)
nxi = saxes.n[0]; oxi = srcx[0]; dxi = srcx[1] - srcx[0]
print("Image grid: nxi=%d oxi=%f dxi=%f"%(nxi,oxi,dxi))

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
  img += wei.image_data(datw,dt,ntx=16,minf=1,maxf=51,vel=velint,nhx=0,nrmax=20,
                        nthrds=40,client=client)

sep.write_file("sigimgw2.H",img,ds=[dz,dy,dxi],os=[oz,0.0,oxi])

# Shutdown dask
client.shutdown()
shutdown_sshcluster(hosts)
