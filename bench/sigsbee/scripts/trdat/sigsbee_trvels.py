import inpout.seppy as seppy
import numpy as np
import zmq
from velocity.veltrchunkr import veltrchunkr
from server.distribute import dstr_collect
from client.sshworkers import launch_sshworkers, kill_sshworkers, create_host_list

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic', 'storm', 'torch', 'jarvis']
wph = [10,10,10,10]
hin = create_host_list(hosts,wph)
cfile = "/homes/sep/joseph29/projects/resfoc/velocity/veltrworker.py"
launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=False)

# Make generator
nchnk = len(hin)
vcnkr = veltrchunkr(nchnk,
                    nmodels=40,
                    nx=2133,ny=20,nz=1201,layer=100,maxvel=3800)
vcnkr.set_ano_pars(minnax=300,maxnax=700,minnaz=100,maxnaz=250)
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
kill_sshworkers(cfile,hosts,verb=False)

