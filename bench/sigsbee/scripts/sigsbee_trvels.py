import inpout.seppy as seppy
import numpy as np
import zmq
from velocity.veltrchunkr import veltrchunkr
from server.distribute import dstr_collect
from client.slurmworkers import launch_slurmworkers, kill_slurmworkers

# IO
sep = seppy.sep()

# Start workers
cfile = "/home/joseph29/projects/resfoc/velocity/veltrworker.py"
logpath = "./log"
wrkrs,status = launch_slurmworkers(cfile,ncore=3,mem=8,nworkers=50,queues=['twohour'],
                                   logpath=logpath,slpbtw=0.5,chkrnng=True)

print("Workers status: ",*status)

# Make generator
nchnk = status.count('R')
vcnkr = veltrchunkr(nchnk,
                    nmodels=500,
                    nx=2133,ny=20,nz=1201,layer=100,maxvel=3800)
gen = iter(vcnkr)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Distribute work to workers and collect results
okeys = ['vel','ref','cnv','lbl','ano']
output = dstr_collect(okeys,nchnk,gen,socket)

ovel = np.concatenate(output['vel'])
oref = np.concatenate(output['ref'])
ocnv = np.concatenate(output['cnv'])
olbl = np.concatenate(output['lbl'])
oano = np.concatenate(output['ano'])

# Write output
dz = 0.00762; dx = 0.01143; ox = 3.05562
sep.write_file('sigsbee_trvels.H',ovel.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('sigsbee_trrefs.H',oref.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('sigsbee_trcnvs.H',ocnv.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('sigsbee_trlbls.H',olbl.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('sigsbee_tranos.H',oano.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])

# Clean up
del ovel; del oref; del ocnv; del olbl; del oano
kill_slurmworkers(wrkrs)

