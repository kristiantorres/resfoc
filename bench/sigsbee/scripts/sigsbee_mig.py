import inpout.seppy as seppy
import numpy as np
import zmq
from oway.imagechunkr import imagechunkr
from server.distribute import dstr_sum
from client.pbsworkers import launch_pbsworkers, kill_pbsworkers

# IO
sep = seppy.sep()

# Start workers
cfile = "/data/sep/joseph29/projects/scaas/oway/imageworker.py"
logpath = "./log"
wrkrs,status = launch_pbsworkers(cfile,nworkers=40,queue='default',
                                   logpath=logpath,slpbtw=0.5,chkrnng=True)
print("Workers status: ",*status)

# Read in data
daxes,dat = sep.read_file("sigsbee_shotflat.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in the velocity model
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

# Imaging grid (same as shots)
nxi = saxes.n[0]; oxi = srcx[0]; dxi = srcx[1] - srcx[0]
print("Image grid: nxi=%d oxi=%f dxi=%f"%(nxi,oxi,dxi))

nchnk = status.count('R')
icnkr = imagechunkr(nchnk,
                    nxi,dxi,ny,dy,nz,dz,velin,
                    dat,dt,minf=1.0,maxf=51.0,
                    nrec=nrec,srcx=srcx,recx=recx,
                    ovx=ovx,dvx=dvx,ox=oxi)

icnkr.set_image_pars(ntx=16,nthrds=16,nrmax=20,sverb=True)
gen = iter(icnkr)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Distribute work to workers and sum over results
img = dstr_sum('cid','result',nchnk,gen,socket,icnkr.get_img_shape())

# Zero-offset image
sep.write_file("sigimg.H",img,os=[oz,0.0,oxi],ds=[dz,1.0,dxi])

# Clean up
kill_pbsworkers(wrkrs)

