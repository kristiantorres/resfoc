import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
from genutils.movie import viewcube3d
import matplotlib.pyplot as plt

# IO
sep = seppy.sep()

# Read in the geometry
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2.H")
naxes,nrec = sep.read_file("/data3/northsea_dutch_f3/f3_nrec2.H")
nrec = nrec.astype('int32')

# Window
nsht = 2
nd = np.sum(nrec[:nsht])
srcxw = srcx[:nsht]
srcyw = srcy[:nsht]
nrecw = nrec[:nsht]
recxw = recx[:nd]
recyw = recy[:nd]

# Read in the data
daxes,dat = sep.read_wind("f3_shots2_muted_debub_onetr.H",fw=0,nw=nd)
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
nt,ntr = daxes.n; ot,_ = daxes.o; dt,_ = daxes.d

# Read in the windowed velocity model
vaxes,vel = sep.read_file("miglintz.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F').T)
ny,nx,nz = vel.shape
dz,dx,dy = vaxes.d; oz,ox,oy = vaxes.o

# Read in in the migration cube
maxes,mig = sep.read_file("/data3/northsea_dutch_f3/mig/mig.T")
mig   = np.ascontiguousarray(mig.reshape(maxes.n,order='F').T)
migw  = mig[5:505,200:1200,:]
migww = migw[25:125,:800,:800]

velw = vel[25:125,:500,:500]
nyw,nxw,nz = velw.shape

oyw = oy + 25*dy

#viewcube3d(migww.T,ds=[0.004,dy,dx],os=[oz,oyw,ox],cmap='gray',width3=1.0)
#viewcube3d(velw.T,ds=[dz,dy,dx],os=[oz,oyw,ox],cmap='jet',width3=1.0,show=False)
#plt.show()

srcxs,srcys = srcxw*0.001,srcyw*0.001
recxs,recys = recxw*0.001,recyw*0.001
wei = geom.coordgeom(nxw,dx,nyw,dy,nz,dz,ox=ox,oy=oyw,srcxs=srcxs,srcys=srcys,
                     nrec=nrecw,recxs=recxs,recys=recys)

velwt = np.ascontiguousarray(np.transpose(velw,(2,0,1)))
migwt = np.ascontiguousarray(np.transpose(migww,(2,0,1)))
wei.plot_acq(migwt,iz=400,srcs=True,cmap='gray',show=False)
wei.plot_acq(velwt,iz=400,recs=True,cmap='jet')

img = wei.image_data(dat,dt,ntx=16,nty=16,minf=1.0,maxf=71.0,vel=velwt,nhx=0,nrmax=20,nthrds=40,wverb=True)
imgt = np.transpose(img,(0,2,1))

#sep.write_file("f3img.H",imgt,ds=[dz,dx,dy],os=[0.0,ox,oyw])

