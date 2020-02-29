import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig, convert2time
import matplotlib.pyplot as plt
from utils.movie import viewframeskey
from utils.signal import butter_bandpass_filter

# Read in the processed fault image
sep = seppy.sep([])
iaxes,img = sep.read_file(None,ifname='fltimgprc.H')
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
rmig = preresmig(eimg,[dh,dx,dz],time=False)

# Conversion to time
#TODO: need to figure out how to get them to map to the same sample positions
time = convert2time(rmig,dz,nt=nz,dt=0.004,vc=8000)

# Visualize the frames
viewframeskey(rmig[:,zoff,:,:],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9,show=False)
viewframeskey(time[:,zoff,:,:],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9,show=True)

