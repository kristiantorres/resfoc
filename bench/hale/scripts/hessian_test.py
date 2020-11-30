import inpout.seppy as seppy
import numpy as np
from oway.hessnchunkr import hessnchunkr
from server.distribute import dstr_sum_adapt
from server.utils import startserver
from scaas.wavelet import ricker
from oway.costaper import costaper
from scaas.trismooth import smooth
from scaas.off2ang import off2angkzx, get_angkzx_axis
from client.slurmworkers import launch_slurmworkers, kill_slurmworkers

# IO
sep = seppy.sep()

# Start workers
cfile = "/home/joseph29/projects/scaas/oway/hessnworker.py"
logpath = "./log"
wrkrs,status = launch_slurmworkers(cfile,nworkers=30,queue=['sep','twohour'],
                                   logpath=logpath,slpbtw=4.0,mode='adapt')
print("Workers status: ",*status)

# Read in velocity
vaxes,vel = sep.read_file("onevel.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F')).astype('float32')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
# Read in reflectivity
raxes,ref = sep.read_file("oneref.H")
[nz,nrx] = raxes.n; [dz,drx] = raxes.d; [oz,orx] = raxes.o
ref = np.ascontiguousarray(ref.reshape(raxes.n,order='F')).astype('float32')
ny = 1; dy = 1

# Read in the acquisition geometry
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

# Create ricker wavelet
n1   = 1500; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav  = ricker(n1,d1,freq,amp,t0)

# Start server
context,socket = startserver()

# Prepare inputs
slo   = np.zeros([nz,ny,nvx],dtype='float32')
refin = np.zeros([nz,ny,nrx],dtype='float32')

# Smooth in slowness
slo[:,0,:] = smooth(1/vel,rect1=40,rect2=30)
vel = 1/slo

# Build the reflectivity
refin[:,0,:] = costaper(ref,nw1=16)
print("Image grid: nxi=%d oxi=%f dxi=%f"%(nrx,orx,drx))

# Lower bound is number of workers
# Upper bound is number of shots
nchnk = len(nrec)//2
hcnkr = hessnchunkr(nchnk,
                    drx,dy,dz,
                    refin,vel,vel,wav,d1,t0,minf=1.0,maxf=51.0,
                    nrec=nrec,srcx=srcx,recx=recx,
                    ovx=ovx,dvx=dvx,ox=orx)

hcnkr.set_hessn_pars(ntx=16,nhx=20,nthrds=48,nrmax=20,mpx=100,sverb=False)
gen = iter(hcnkr)

# Distribute work to workers and sum over results
img = dstr_sum_adapt('cid','result',nchnk,gen,socket,hcnkr.get_img_shape(),
                     wrkrs,verb=True)

# Transpose the image
imgt = np.ascontiguousarray(np.transpose(img,(0,1,3,4,2))) # [nhy,nhx,nz,ny,nx] -> [nhy,nhx,ny,nx,nz]

# Get offset axis
nhx,ohx,dhx = hcnkr.get_offx_axis()

sep.write_file("hessian_test.H",imgt.T,os=[oz,orx,0.0,ohx],ds=[dz,drx,1.0,dhx])

# Clean up
kill_slurmworkers(wrkrs)

