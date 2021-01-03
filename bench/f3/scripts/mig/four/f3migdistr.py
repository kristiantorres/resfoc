import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
from oway.imagechunkr import imagechunkr
from server.distribute import dstr_sum
from server.utils import startserver
from client.sshworkers import launch_sshworkers, kill_sshworkers
from seis.f3utils import plot_acq
from genutils.movie import viewcube3d

# IO
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','torch','thing','jarvis']
cfile = "/homes/sep/joseph29/projects/scaas/oway/imageworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1,clean=True)

# Read in the geometry
sxaxes,srcx = sep.read_file("f3_srcx3_full_clean2.H")
syaxes,srcy = sep.read_file("f3_srcy3_full_clean2.H")
rxaxes,recx = sep.read_file("f3_recx3_full_clean2.H")
ryaxes,recy = sep.read_file("f3_recy3_full_clean2.H")
naxes,nrec = sep.read_file("f3_nrec3_full_clean2.H")
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
daxes,dat = sep.read_wind("f3_shots3interp_full_debub_onetr_gncw.H",fw=0,nw=nd)
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
nt,ntr = daxes.n; ot,_ = daxes.o; dt,_ = daxes.d

# Read in the windowed velocity model
vaxes,vel = sep.read_file("miglintz5m.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F').T)
ny,nx,nz = vel.shape
dz,dx,dy = vaxes.d; oz,ox,oy = vaxes.o

velw = vel[:460,:500,:1000]
nyw,nxw,nz = velw.shape
oyw = oy + 0*dy
velwt = np.ascontiguousarray(np.transpose(velw,(2,0,1)))

# Read in time slice for QC
saxes,slc = sep.read_wind("migwt.T",fw=400,nw=1)
slc = slc.reshape(saxes.n,order='F')
slcw = slc[:,:1000]

srcxs,srcys = srcxw*0.001,srcyw*0.001
recxs,recys = recxw*0.001,recyw*0.001

plot_acq(srcxs,srcys,recxs,recys,slcw,ox=ox,oy=oyw,recs=False,show=True)

nchnk = len(hosts)
icnkr = imagechunkr(nchnk,
                    nxw,dx,nyw,dy,nz,dz,velwt,
                    dat,dt,minf=1.0,maxf=61.0,
                    nrec=nrecw,srcx=srcxs,recx=recxs,
                    srcy=srcys,recy=recys,ox=ox,oy=oy)
icnkr.set_image_pars(ntx=16,nty=16,nhx=0,nrmax=20,nthrds=40,sverb=True,wverb=True)
gen = iter(icnkr)

print(velwt.shape,oz,ox,oyw)
viewcube3d(velwt,ds=[dz,dx,dy],os=[oz,ox,oyw],cmap='jet')
# Bind to socket
context,socket = startserver()

# Distribute work to workers and sum the results
img = dstr_sum('cid','result',nchnk,gen,socket,icnkr.get_img_shape())
imgt = np.transpose(img,(0,2,1))

# Zero offset image
sep.write_file("f3imgdistr_fullxl.H",imgt,ds=[dz,dx,dy],os=[0.0,ox,oyw])

kill_sshworkers(cfile,hosts,verb=False)

