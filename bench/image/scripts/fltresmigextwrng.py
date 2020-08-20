import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig,convert2time,get_rho_axis
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

# Read in the processed fault image
sep = seppy.sep()
iaxes,img = sep.read_file('fltimgbigwrng.H')
img = img.reshape(iaxes.n,order='F')

imgt = np.ascontiguousarray(img.T).astype('float32')

# Get axes
[nz,nx,nh] = iaxes.n; [dz,dx,dh] = iaxes.d; [oz,ox,oh] = iaxes.o

# Depth Residual migration
inro = 17; idro = 0.00125
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)
rmig = preresmig(imgt,[dh,dx,dz],nps=[2049,1025,513],nro=inro,dro=idro,time=False,nthreads=24,verb=True)

# Conversion to time
#time = convert2time(rmig,dz,dt=0.004,dro=odro,verb=True)

# Visualize the frames
#viewimgframeskey(rmig[:,16,:,:],ttlstring='rho=%.2f',ottl=ooro,dttl=odro,wbox=14,hbox=7,pclip=0.9,show=False)
#viewimgframeskey(time[:,16,:,:],ttlstring='rho=%.2f',ottl=ooro,dttl=odro,wbox=14,hbox=7,pclip=0.9,show=True)

# Write the subsurface offsets to file
sep.write_file("fltimgbigreswrng.H",rmig.T,ds=[dz,dx,dh,odro],os=[0,0,oh,ooro])

