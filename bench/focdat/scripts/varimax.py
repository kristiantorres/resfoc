import numpy as np
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.resmig import preresmig,get_rho_axis,convert2time
from deeplearn.focuslabels import varimax
import matplotlib.pyplot as plt
from genutils.movie import viewimgframeskey
import inpout.seppy as seppy

nx = 400; dx = 10
nz = 200; dz = 10
nh = 16;  dh = 10

#vel = np.zeros([nz,nx],dtype='float32')
#ref = np.zeros([nz,nx],dtype='float32')
#
#vel[:] = 2000.0
#ref[99,199] = 1.0
#
## Acquisition geometry
#dsx = 10; bx = 50; bz = 50
#prp = geom.defaultgeom(nx,dx,nz,dz,nsx=41,dsx=dsx,bx=bx,bz=bz)
#
#prp.plot_acq(vel,cmap='jet',show=True)
#
## Create data axes
#ntu = 3000; dtu = 0.001;
#freq = 20; amp = 100.0; dly = 0.2;
#wav = ricker(ntu,dtu,freq,amp,dly)
#
## Model true linearized data
#dtd = 0.004
#allshot = prp.model_lindata(vel,ref,wav,dtd,verb=True,nthrds=24)
#
## Wave equation depth migration
#img = prp.wem(vel,allshot,wav,dtd,nh=nh,verb=True,nthrds=24)
#
#imgt = np.transpose(img,(0,2,1)) # [nh,nz,nx] -> [nh,nx,nz]
#
## Residual migration
#inro = 21; idro = 0.00125
#rmig = preresmig(imgt,[dh,dx,dz],nps=[1025,nx+1,nz+1],nro=inro,dro=idro,time=False,nthreads=24,verb=True)
#onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)
#
#stk = rmig[:,nh,:,:]

sep = seppy.sep()
#sep.write_file("ptscatstk.H",stk.T,ds=[dz,dx,odro],os=[0,0,ooro])

saxes,stk = sep.read_file("ptscatstk.H")
stk = np.ascontiguousarray(stk.reshape(saxes.n,order='F').T)
[nz,nx,nro] = saxes.n; [dz,dx,dro] = saxes.d; [oz,ox,oro] = saxes.o

#aaxes,ang = sep.read_file("ptscatang.H")
#ang = np.ascontiguousarray(ang.reshape(aaxes.n,order='F').T)
#[nz,na,nx,nro] = aaxes.n; [dz,da,dx,dro] = aaxes.d; [oz,oa,ox,oro] = aaxes.o
#stk = np.sum(ang,axis=2)

winsize = 32
begz = 99  - winsize; endz = 99  + winsize
begx = 199 - winsize; endx = 199 + winsize
ptch = stk[:,begx:endx,begz:endz]
nzp = ptch.shape[2]; nxp = ptch.shape[1]

viewimgframeskey(ptch,pclip=0.8,interp='sinc')

norm = np.zeros(nro)
for iro in range(nro):
  norm[iro] = varimax(ptch[iro])
  print("rho=%.4f,varimax=%.2f"%(iro*dro + oro,norm[iro]))

rho = np.linspace(oro,oro+(nro-1)*dro,nro)
plt.plot(rho,norm); plt.show()
