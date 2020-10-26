import inpout.seppy as seppy
import numpy as np
import zmq
from oway.hessnchunkr import hessnchunkr
from server.distribute import dstr_sum
from scaas.wavelet import ricker
from oway.costaper import costaper
from scaas.trismooth import smooth
from client.sshworkers import launch_sshworkers, kill_sshworkers

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','thing','storm','torch','jarvis']
cfile = "/homes/sep/joseph29/projects/scaas/oway/hessnworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Read in the velocity and reflectivity models
vaxes,vels = sep.read_file("hale_trvels.H")
vels = np.ascontiguousarray(vels.reshape(vaxes.n,order='F')).astype('float32')
raxes,refs = sep.read_file("hale_trrefs.H")
refs = np.ascontiguousarray(refs.reshape(raxes.n,order='F')).astype('float32')
[nz,nx,nm] = vaxes.n; [dz,dx,dm] = vaxes.d; [oz,ox,om] = vaxes.o
ny = 1; dy = 1

## Read in the acquisition geometry
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

# Convert velocity to slowness
slo = np.zeros([nz,ny,nx],dtype='float32')
ref = np.zeros([nz,ny,nx],dtype='float32')

# Create ricker wavelet
n1   = 1500; d1 = 0.004;
freq = 20; amp = 0.5; t0 = 0.2;
wav  = ricker(n1,d1,freq,amp,t0)

# Output imgs
nhx = 20; nhy = 0
imgts = np.zeros([nm,2*nhy+1,2*nhx+1,ny,nx,nz],dtype='float32')

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

nm = vels.shape[-1]

for im in range(nm):
  print("Model=%d"%(im))
  # Get the current example
  velin = vels[:,:,im]
  refin = refs[:,:,im]

  # Smooth in slowness
  slo[:,0,:] = smooth(1/velin,rect1=35,rect2=35)
  vel = 1/slo

  # Build the reflectivity
  reftap = costaper(refin,nw1=16)
  ref[:,0,:] = reftap

  nchnk = len(hosts)
  hcnkr = hessnchunkr(nchnk,
                      dx,dy,dz,
                      ref,vel,vel,wav,d1,t0,minf=1.0,maxf=51.0,
                      nrec=nrec,srcx=srcx,recx=recx,ox=ox)

  hcnkr.set_hessn_pars(ntx=16,nhx=20,nthrds=40,nrmax=20,mpx=100,sverb=True)
  gen = iter(hcnkr)

  # Distribute work to workers and sum over results
  img = dstr_sum('cid','result',nchnk,gen,socket,hcnkr.get_img_shape())

  # Transpose the image
  imgts[im] = np.transpose(img,(0,1,3,4,2)) # [nhy,nhx,nz,ny,nx] -> [nhy,nhx,ny,nx,nz]

# Get offset axis
nhx,ohx,dhx = hcnkr.get_offx_axis()
sep.write_file("hale_trimgs.H",imgts.T,os=[oz,ox,0.0,ohx,0.0,0.0],ds=[dz,dx,dy,dhx,1.0,1.0])

kill_sshworkers(cfile,hosts,verb=False)

