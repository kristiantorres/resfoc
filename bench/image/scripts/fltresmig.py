import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig, convert2time
import matplotlib.pyplot as plt
from genutils.movie import viewframeskey
from genutils.signal import butter_bandpass_filter

# Read in the processed fault image
sep = seppy.sep([])
iaxes,img = sep.read_file(None,ifname='fltimgwrng3prc.H')
img = img.reshape(iaxes.n,order='F')

# Get axes
nz = iaxes.n[0]; oz = iaxes.o[0]; dz = iaxes.d[0]
nx = iaxes.n[1]; ox = iaxes.o[1]; dx = iaxes.d[1]

# Make axes
nh  = 10 ; oh  = -200;  dh  = dx
nro = 6  ; oro = 1.0 ;  dro = 0.01
rnh = 2*nh + 1; zoff = 10

eimg = np.zeros([rnh,nx,nz],dtype='float32')
eimg[zoff,:,:] = img.T

# Depth Residual migration
rmig = preresmig(eimg,[dh,dx,dz],time=False,nthreads=2*nro-1)

# Conversion to time
time = convert2time(rmig,dz,dt=0.004)

# Visualize the frames
#viewframeskey(rmig[:,zoff,:,:],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9,show=False)
#viewframeskey(time[:,zoff,:,:],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9,show=True)

# Write the files
raxes = seppy.axes([nz,nx,2*nro-1],[oz,ox,oro-dro*(nro-1)],[dz,dx,dro])
sep.write_file(None,raxes,rmig[:,zoff,:,:].T,ofname='drmigwrng.H')
sep.write_file(None,raxes,time[:,zoff,:,:].T,ofname='trmigwrng.H')

