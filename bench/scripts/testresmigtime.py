import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import preresmig, rhoaxis, convert2time
import matplotlib.pyplot as plt
from utils.movie import viewframeskey
from utils.signal import butter_bandpass_filter

## Create point scatterer model
nz  = 256; oz  =  0.0;  dz  = 20.0
nx  = 400; ox  =  0.0;  dx  = 20.0
nh  = 10 ; oh  = -200;  dh  = dx
nro = 6  ; oro = 1.0 ;  dro = 0.01

rnh = 2*nh + 1; zoff = 10
iaxes = seppy.axes([rnh,nx,nz],[-dx*nh,0.0,0.0],[dx,dx,dz])
taxes = seppy.axes([nz,nx,rnh],[0.0,0.0,-dx*nh],[dz,dx,dx])

pt = np.zeros([rnh,nx,nz],dtype='float32')
pt[zoff,200,128] = 1.0

img = np.zeros(pt.shape,dtype='float32')
img[zoff,:,:] = np.array([butter_bandpass_filter(pt[zoff,ix,:].T,0.002,0.015,1/dz) for ix in range(nx)])

#rmig = preresmig(img,[dh,dx,dz])
raxis = rhoaxis()

#sep = seppy.sep([])
#raxes = seppy.axes([nz,nx,rnh,raxis['fnro']],[0.0,0.0,-dx*nh,raxis['foro']],[dz,dx,dx,raxis['fdro']])
#sep.write_file(None,raxes,rmig.T,ofname='newres.H')
sep = seppy.sep([])
raxes,rmig = sep.read_file(None,'newres.H')
rmig = rmig.reshape(raxes.n,order='F')
rmig = np.ascontiguousarray(rmig.T)

time = convert2time(rnh,nx,nz,dz,nz,0.004,rmig,vc=6000)

print(np.min(time))

#viewframeskey(rmig[:,zoff,:,:],ttlstring='rho=%.2f',ottl=raxis['foro'],dttl=raxis['fdro'],wbox=14,hbox=7,pclip=0.9)
viewframeskey(time[:,zoff,:,:],ttlstring='rho=%.2f',ottl=raxis['foro'],dttl=raxis['fdro'],wbox=14,hbox=7,pclip=0.9)

# Convert them to time
#rmigtime = np.zeros(rmig.shape,dtype='float32')
#vel = np.zeros(pt.shape,dtype='float32')
#for iro in range(fnro):
#  rho = foro + iro*dro
#  print(rho)
#  vel[:] = rho/2000.0
#  rst.convert2time(nz,0.0,0.004,vel,rmigiftswind,rmigtime)

#viewframeskey(rmigtime[:,10,:,:],ttlstring='rho=%.2f',ottl=foro,dttl=dro,wbox=14,hbox=7,pclip=0.9)
