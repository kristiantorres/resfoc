import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig, convert2time
import matplotlib.pyplot as plt
from genutils.movie import viewframeskey
from genutils.signal import butter_bandpass_filter

## Create point scatterer model
nz  = 256; oz  =  0.0;  dz  = 20.0
nx  = 400; ox  =  0.0;  dx  = 20.0
nh  = 10 ; oh  = -200;  dh  = dx
nro = 6  ; oro = 1.0 ;  dro = 0.01

rnh = 2*nh + 1; zoff = 10

# Make a spike
pt = np.zeros([rnh,nx,nz],dtype='float32')
pt[zoff,200,128] = 1.0

# Make it bandlimited
img = np.zeros(pt.shape,dtype='float32')
img[zoff,:,:] = np.array([butter_bandpass_filter(pt[zoff,ix,:].T,0.002,0.015,1/dz) for ix in range(nx)])

# Depth Residual migration
rmig = preresmig(img,[dh,dx,dz],time=False)

# Conversion to time
time = convert2time(rmig,dz,dt=0.004)

# Visualize the frames
viewframeskey(rmig[:,zoff,:,:],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9,show=False)
viewframeskey(time[:,zoff,:,:nz],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9)
