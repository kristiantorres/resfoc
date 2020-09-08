import inpout.seppy as seppy
import numpy as np
import zmq
from oway.hessnchunkr import hessnchunkr
from server.distribute import dstr_sum
from scaas.wavelet import ricker
from oway.costaper import costaper
from scaas.trismooth import smooth
from scaas.off2ang import off2angkzx, get_angkzx_axis
from client.pbsworkers import launch_pbsworkers, kill_pbsworkers

# IO
sep = seppy.sep()

# Start workers
cfile = "/data/sep/joseph29/projects/scaas/oway/hessnworker.py"
logpath = "./log"
wrkrs,status = launch_pbsworkers(cfile,nworkers=50,queue='default',
                                   logpath=logpath,slpbtw=0.5,chkrnng=True)
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

# Read in the acquisition geometry
saxes,srcx = sep.read_file("sigsbee_srcxflat.H")
raxes,recx = sep.read_file("sigsbee_recxflat.H")
_,nrec= sep.read_file("sigsbee_nrec.H")
nrec = nrec.astype('int')

# Create ricker wavelet
n1   = 1500; d1 = 0.008;
freq = 20; amp = 0.5; t0 = 0.2;
wav  = ricker(n1,d1,freq,amp,t0)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Prepare inputs
velin = np.zeros([nvz,ny,nvx],dtype='float32')
refin = np.zeros([nrz,ny,nrx],dtype='float32')

velin[:,0,:] = vel[:]

# Build the reflectivity
reftap = costaper(ref,nw1=16)
refin[:,0,:] = reftap[:]
print("Image grid: nxi=%d oxi=%f dxi=%f"%(nrx,orx,drx))

nchnk = status.count('R')
hcnkr = hessnchunkr(nchnk,
                    drx,dy,drz,
                    refin,velin,velin,wav,d1,t0,minf=1.0,maxf=51.0,
                    nrec=nrec,srcx=srcx,recx=recx,
                    ovx=ovx,dvx=dvx,ox=orx)

hcnkr.set_hessn_pars(ntx=16,nhx=20,nthrds=16,nrmax=20,mpx=100,sverb=True)
gen = iter(hcnkr)

# Distribute work to workers and sum over results
img = dstr_sum('cid','result',nchnk,gen,socket,hcnkr.get_img_shape())

# Transpose the image
imgt = np.ascontiguousarray(np.transpose(img,(0,1,3,4,2))) # [nhy,nhx,nz,ny,nx] -> [nhy,nhx,ny,nx,nz]

# Convert to angle
na = 64
nhx,ohx,dhx = hcnkr.get_offx_axis()
iang = off2angkzx(imgt,ohx,dhx,drz,na=na,nthrds=10,transp=True,cverb=False)
na,oa,da = get_angkzx_axis(na,amax=60)

# Get offset axis
sep.write_file("sigsbee_hessang_test.H",iang.T,os=[orz,oa,0.0,orx,0.0],ds=[drz,da,1.0,drx,1.0])

# Clean up
kill_pbsworkers(wrkrs)

