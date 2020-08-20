import inpout.seppy as seppy
import numpy as np
import zmq
from oway.imagechunkr import imagechunkr
from server.distribute import dstr_collect, dstr_sum
from client.sshworkers import launch_sshworkers, kill_sshworkers
import matplotlib.pyplot as plt

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm', 'thing']
cfile = "/homes/sep/joseph29/projects/scaas/oway/imagesshworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=False)

# Read in data
daxes,dat = sep.read_file("hale_shotflatbob.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in the velocity model
vaxes,vel = sep.read_file("vintzcomb.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

nxi = int(2*nvx); dxi = dvx/2; oxi = ovx

nchnk = len(hosts)
icnkr = imagechunkr(nchnk,
                    nxi,dxi,ny,dy,nz,dz,velin,
                    dat,dt,minf=1.0,maxf=51.0,
                    nrec=nrec,srcx=srcx,recx=recx,
                    ovx=ovx,dvx=dvx,ox=oxi)

icnkr.set_image_pars(ntx=16,nhx=20,nthrds=40,nrmax=10,sverb=False)
gen = iter(icnkr)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Distribute work to workers and sum over results
img = dstr_sum('cid','result',nchnk,gen,socket,icnkr.get_img_shape())

# Transpose the image
imgt = np.transpose(img,(2,4,3,1,0)) # [nhy,nhx,nz,ny,nx] -> [nz,nx,ny,nhx,nhy]
# Get offset axis
nhx,ohx,dhx = icnkr.get_offx_axis()
sep.write_file("spimgextbobdistr.H",imgt,os=[oz,oxi,0.0,ohx,0.0],ds=[dz,dxi,dy,dhx,1.0])

kill_sshworkers(cfile,hosts,verb=False)

