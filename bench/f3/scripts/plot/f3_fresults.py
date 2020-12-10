import inpout.seppy as seppy
import numpy as np
from seis.f3utils import plot_acq
from resfoc.gain import agc
from genutils.movie import viewcube3d

sep = seppy.sep()

# Read in the geometry
sxaxes,srcx = sep.read_file("/data3/northsea_dutch_f3/f3_srcx2.H")
syaxes,srcy = sep.read_file("/data3/northsea_dutch_f3/f3_srcy2.H")
rxaxes,recx = sep.read_file("/data3/northsea_dutch_f3/f3_recx2.H")
ryaxes,recy = sep.read_file("/data3/northsea_dutch_f3/f3_recy2.H")

srcx *= 0.001
srcy *= 0.001
recx *= 0.001
recy *= 0.001

# Read in time slice for QC
saxes,slc = sep.read_wind("migwt.T",fw=400,nw=1)
dy,dx,dt = saxes.d; oy,ox,ot = saxes.o
slc = slc.reshape(saxes.n,order='F')
slcw = slc[25:125,:500]

# Plot the acquisition
plot_acq(srcx,srcy,recx,recy,slc,ox=ox,oy=oy,recs=False,show=True)

# Read in velocity model
vaxes,vel = sep.read_file("miglintz.H")
dz,dx,dy = vaxes.d; oz,ox,oy = vaxes.o
vel = vel.reshape(vaxes.n,order='F')
velw = vel[:500,:500,25:125]

# Read in image
iaxes,img = sep.read_file("f3img2-12.H")
img = img.reshape(iaxes.n,order='F')
dz,dx,dy = iaxes.d; oz,ox,oy = iaxes.o
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
gimg = agc(img)


ox,oy = 0,0
# Plot the velocity model
viewcube3d(velw,ds=[dz,dy,dx],os=[oz,oy,ox],cmap='jet',interp='bilinear',cbar=True,
           label1='X (km)',label2='Y (km)',label3='Z (km)',width3=1.0,cbarlabel='km/s',show=False)

# Taper the top of the image
viewcube3d(gimg.T,ds=[dz,dx,dy],os=[oz,oy,ox],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',pclip=0.1,width3=1.0)
