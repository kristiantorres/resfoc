import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig, convert2time
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey
from utils.signal import butter_bandpass_filter

# Read in the processed fault image
sep = seppy.sep([])
iaxes,img = sep.read_file(None,ifname='fltimgextprc.H')
img = img.reshape(iaxes.n,order='F')

imgt = np.ascontiguousarray(np.transpose(img,(2,1,0)))

# Get axes
nz  = iaxes.n[0]; oz = iaxes.o[0]; dz = iaxes.d[0]
nx  = iaxes.n[1]; ox = iaxes.o[1]; dx = iaxes.d[1]
rnh = iaxes.n[2]; oh = iaxes.o[2]; dh = iaxes.d[2]

# Make axes
nh  = 10 ; oh  = -200;  dh  = dx
nro = 6  ; oro = 1.0 ;  dro = 0.01
zoff = 10

# Depth Residual migration
rmig = preresmig(imgt,[dh,dx,dz],time=False,nthreads=2*nro-1)

print(rmig.shape)

# Conversion to time
time = convert2time(rmig,dz,dt=0.004)

# Visualize the frames
viewimgframeskey(rmig[:,zoff,:,:],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9,show=False)
viewimgframeskey(time[:,zoff,:,:],ttlstring='rho=%.2f',ottl=oro-dro*(nro-1),dttl=dro,wbox=14,hbox=7,pclip=0.9,show=True)

