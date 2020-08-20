import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig,convert2time,get_rho_axis
from scaas.off2ang import off2ang,get_ang_axis
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey

# Read in the processed fault image
sep = seppy.sep()
iaxes,img = sep.read_file('fltimgbig.H')
img = img.reshape(iaxes.n,order='F')

imgt = np.ascontiguousarray(img.T).astype('float32')

# Get axes
[nz,nx,nh] = iaxes.n; [dz,dx,dh] = iaxes.d; [oz,ox,oh] = iaxes.o

# Window the data only over the faults
f2 = 250; n2 = 512
imgtwind = imgt[:,f2:f2+n2,:]

# Depth Residual migration
inro = 17; idro = 0.00125
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)
rmig = preresmig(imgtwind,[dh,dx,dz],nps=[2049,513,513],nro=inro,dro=idro,time=False,nthreads=24,verb=True)

# Conversion to time
time = convert2time(rmig,dz,dt=0.004,dro=odro,verb=True)

# Visualize the frames
viewimgframeskey(rmig[:,16,:,:],ttlstring='rho=%.2f',ottl=ooro,dttl=odro,wbox=14,hbox=7,pclip=0.9,show=False)
viewimgframeskey(time[:,16,:,:],ttlstring='rho=%.2f',ottl=ooro,dttl=odro,wbox=14,hbox=7,pclip=0.9,show=True)

# Write the subsurface offsets to file
sep.write_file("fltimgbigreswind.H",rmig.T,ds=[dz,dx,dh,odro],os=[0,0,oh,ooro])

# Convert to angle
timet = np.ascontiguousarray(np.transpose(time,(0,1,3,2))).astype('float32')
stormang = off2ang(timet,oh,dh,dz,oro=ooro,dro=odro,transp=True,verb=True,nthrds=24)
na,oa,da = get_ang_axis()
# Write to file
stormangt = np.transpose(stormang,(2,1,3,0))
sep.write_file("fltimgresangwind.H",stormangt,os=[oz,oa,ox,ooro],ds=[dz,da,dx,odro])

