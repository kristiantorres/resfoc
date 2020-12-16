import os
import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
from oway.imagechunkrio import imagechunkrio, id_generator
from server.distribute import dstr_collect
from server.utils import startserver
from client.sshworkers import launch_sshworkers, kill_sshworkers
from seis.f3utils import sum_extimgs
from genutils.movie import viewcube3d
import matplotlib.pyplot as plt

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','torch','thing','jarvis']
cfile = "/homes/sep/joseph29/projects/scaas/oway/imageworkerio.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Read in the geometry
sxaxes,srcx = sep.read_file("/net/brick5/data3/northsea_dutch_f3/windowed_data/f3_srcx2.H")
syaxes,srcy = sep.read_file("/net/brick5/data3/northsea_dutch_f3/windowed_data/f3_srcy2.H")
rxaxes,recx = sep.read_file("/net/brick5/data3/northsea_dutch_f3/windowed_data/f3_recx2.H")
ryaxes,recy = sep.read_file("/net/brick5/data3/northsea_dutch_f3/windowed_data/f3_recy2.H")
naxes,nrec = sep.read_file("/net/brick5/data3/northsea_dutch_f3/windowed_data/f3_nrec2.H")
nrec = nrec.astype('int32')

# Window
nsht = 5
nd = np.sum(nrec[:nsht])
srcxw = srcx[:nsht]
srcyw = srcy[:nsht]
nrecw = nrec[:nsht]
recxw = recx[:nd]
recyw = recy[:nd]

# Read in the data
daxes,dat = sep.read_wind("f3_shots2interp_muted_debub_onetr_gncw.H",fw=0,nw=nd)
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
nt,ntr = daxes.n; ot,_ = daxes.o; dt,_ = daxes.d

# Read in the windowed velocity model
vaxes,vel = sep.read_file("miglintz5m.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F').T)
ny,nx,nz = vel.shape
dz,dx,dy = vaxes.d; oz,ox,oy = vaxes.o

velw = vel[25:125,:500,:1000]
nyw,nxw,nz = velw.shape
oyw = oy + 25*dy
velwt = np.ascontiguousarray(np.transpose(velw,(2,0,1)))

srcxs,srcys = srcxw*0.001,srcyw*0.001
recxs,recys = recxw*0.001,recyw*0.001

# Make temporary directory
tag = id_generator()
bdir = "/homes/sep/joseph29/projects/resfoc/bench/f3/f3ext-" + tag + '/'
os.mkdir(bdir)

nchnk = len(hosts)
icnkr = imagechunkrio(nchnk,
                      nxw,dx,nyw,dy,nz,dz,velwt,
                      dat,dt,minf=1.0,maxf=61.0,
                      nrec=nrecw,srcx=srcxs,recx=recxs,
                      srcy=srcys,recy=recys,ox=ox,oy=oy,
                      bname='f3extimg',bdir=bdir)
icnkr.set_image_pars(ntx=16,nty=16,nhx=20,nrmax=20,nthrds=40,sverb=True,wverb=True)
gen = iter(icnkr)

viewcube3d(velwt,ds=[dz,dx,dy],os=[oz,ox,oyw],cmap='jet',width1=1.0)
# Bind to socket
context,socket = startserver()

# Distribute the work to the workers and collect the file names
okeys = ['ofname']
odict = dstr_collect(okeys,nchnk,gen,socket)

ofile = "/homes/sep/joseph29/projects/resfoc/bench/f3/f3imgdistrextio_5m.H"
sum_extimgs(bdir,ofile)

kill_sshworkers(cfile,hosts,verb=False)

