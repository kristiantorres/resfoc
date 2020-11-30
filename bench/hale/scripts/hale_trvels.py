import inpout.seppy as seppy
import numpy as np
from velocity.hale_veltrchunkr import hale_veltrchunkr
from server.distribute import dstr_collect_adapt
from server.utils import startserver
from client.slurmworkers import launch_slurmworkers, kill_slurmworkers, trim_tsworkers

# IO
sep = seppy.sep()

# Read in the velocity model
vaxes,hvel = sep.read_file("vintzcomb.H",form='native')
nvz,nvx = vaxes.n; ovz,ovx = vaxes.o; dvz,dvx = vaxes.d
hvel = hvel.reshape(vaxes.n,order='F').T
vzin = hvel[150,45:]

# Start workers
cfile = "/home/joseph29/projects/resfoc/velocity/hale_veltrworker.py"
logpath = "./log"
nworkers = 50
wrkrs,status = launch_slurmworkers(cfile,ncore=3,mem=8,nworkers=nworkers,queue=['sep','twohour'],
                                   block=['maz132'],logpath=logpath,slpbtw=4.0,mode='adapt')

print("Workers status: ",*status)

# Make generator
nmodels = 500; nchnk = nworkers
vcnkr = hale_veltrchunkr(nchnk,nmodels=nmodels,vzin=vzin)
gen = iter(vcnkr)

# Start server
context,socket = startserver()

# Distribute work to workers and collect results
okeys = ['vel','ref','lbl','ano']
output = dstr_collect_adapt(okeys,nchnk,gen,socket,wrkrs,verb=True)

ovel = np.concatenate(output['vel'])
oref = np.concatenate(output['ref'])
olbl = np.concatenate(output['lbl'])
oano = np.concatenate(output['ano'])

del output

# Write output
dz = 0.005; dx = 0.01675; ox = 5.36
sep.write_file('hale_trvels.H',ovel.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0]); del ovel
sep.write_file('hale_trrefs.H',oref.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0]); del oref
sep.write_file('hale_trlbls.H',olbl.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0]); del olbl
sep.write_file('hale_tranos.H',oano.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0]); del oano

# Clean up
kill_slurmworkers(wrkrs)

