import inpout.seppy as seppy
import numpy as np
from velocity.hale_veltrchunkr import hale_veltrchunkr
from server.utils import startserver
from server.distribute import dstr_collect
from client.sshworkers import launch_sshworkers, kill_sshworkers, create_host_list

# IO
sep = seppy.sep()

# Read in the velocity model
vaxes,hvel = sep.read_file("vintzcomb.H")
nvz,nvx = vaxes.n; ovz,ovx = vaxes.o; dvz,dvx = vaxes.d
hvel = hvel.reshape(vaxes.n,order='F').T
vzin = hvel[150,45:]

# Start workers
hosts = ['fantastic', 'storm', 'torch', 'jarvis', 'thing']
wph = [10]*len(hosts)
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/velocity/hale_veltrworker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=True)

# Make generator
nchnk = len(hin)
vcnkr = hale_veltrchunkr(nchnk,nmodels=500,vzin=vzin)
gen = iter(vcnkr)

# Bind to socket
cts,socket = startserver()

# Distribute work to workers and collect results
okeys = ['vel','ref','lbl','ano']
output = dstr_collect(okeys,nchnk,gen,socket)

ovel = np.concatenate(output['vel'])
oref = np.concatenate(output['ref'])
olbl = np.concatenate(output['lbl'])
oano = np.concatenate(output['ano'])

# Write output
dz = 0.005; dx = 0.01675; ox = 5.36
sep.write_file('hale_trvels.H',ovel.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('hale_trrefs.H',oref.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('hale_trlbls.H',olbl.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])
sep.write_file('hale_tranos.H',oano.T,os=[0.0,ox,0.0],ds=[dz,dx,1.0])

# Clean up
kill_sshworkers(cfile,hosts,verb=False)

