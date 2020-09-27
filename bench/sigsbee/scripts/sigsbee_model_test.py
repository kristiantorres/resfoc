import inpout.seppy as seppy
import numpy as np
import zmq
from oway.modelchunkr import modelchunkr
from server.distribute import dstr_collect
from scaas.wavelet import ricker
from oway.costaper import costaper
from scaas.trismooth import smooth
from client.slurmworkers import launch_slurmworkers, kill_slurmworkers

# IO
sep = seppy.sep()

# Start workers
cfile = "/home/joseph29/projects/scaas/oway/modelworker.py"
logpath = "./log"
wrkrs,status = launch_slurmworkers(cfile,nworkers=10,queues=['sep'],
                                   logpath=logpath,slpbtw=0.5,
                                   chkrnng=True,mode='quiet')

print("Workers status: ",*status)

# Read in velocity
vaxes,vel = sep.read_file("sigsbee_vel.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F')).astype('float32')
[nvz,nvx] = vaxes.n; [dvz,dvx] = vaxes.d; [ovz,ovx] = vaxes.o
# Read in reflectivity
raxes,ref = sep.read_file("sigsbee_ref.H")
[nrz,nrx] = raxes.n; [drz,drx] = raxes.d; [orz,orx] = raxes.o
ref = np.ascontiguousarray(ref.reshape(raxes.n,order='F')).astype('float32')
ny = 1; dy = 1

## Read in the acquisition geometry
saxes,srcx = sep.read_file("sigsbee_srcxflat.H")
raxes,recx = sep.read_file("sigsbee_recxflat.H")
_,nrec= sep.read_file("sigsbee_nrec.H")
nrec = nrec.astype('int')

# Convert velocity to slowness
velin = np.zeros([nvz,ny,nvx],dtype='float32')
refin = np.zeros([nrz,ny,nrx],dtype='float32')

# Create ricker wavelet
n1   = 1500; d1 = 0.008;
freq = 20; amp = 0.5; t0 = 0.2;
wav  = ricker(n1,d1,freq,amp,t0)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Smooth in slowness
velin[:,0,:] = vel[:]

# Build the reflectivity
reftap = costaper(ref,nw1=16)
refin[:,0,:] = reftap[:]

nchnk = status.count('R')
mcnkr = modelchunkr(nchnk,
                    drx,dy,drz,
                    refin,velin,wav,d1,minf=1.0,maxf=51.0,
                    nrec=nrec,srcx=srcx,recx=recx,
                    ovx=ovx,dvx=dvx,ox=orx)

mcnkr.set_model_pars(ntx=16,nrmax=20,px=100,nthrds=40,sverb=True)
gen = iter(mcnkr)

# Distribute work to workers
okeys = ['result','cid']
output = dstr_collect(okeys,nchnk,gen,socket)

odat = mcnkr.reconstruct_data(output,t0,reg=False,time=True)

# Write out data
sep.write_file("sigsbee_bdat.H",odat.T,ds=[d1,1.0])

# Clean up
kill_slurmworkers(wrkrs)
