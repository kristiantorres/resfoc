import inpout.seppy as seppy
import numpy as np
from scaas.velocity import create_randomptbs_loc, create_randomptb_loc, create_constptb_loc
from utils.plot import plot_imgvelptb
from deeplearn.utils import resample
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in velocity
vaxes,vel = sep.read_file("sigsbee_vel.H")
[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o
dz *= 1000.0; dx *= 1000.0
vel = vel.reshape(vaxes.n,order='F')
vmin = np.min(vel); vmax = np.max(vel)

# Read in reflectivity
raxes,ref = sep.read_file("./dat/ref.H",form='native')
ref = ref.reshape(raxes.n,order='F')
refre = resample(ref,[nz,nx])

# Read in good perturbation
daxes,diff = sep.read_file("overwdiff.H")
diff = diff.reshape(daxes.n,order='F')

# Read in good velocity error
vaxes,velw = sep.read_file("sigsbee_veloverw.H")
velw = velw.reshape(vaxes.n,order='F')

plot_imgvelptb(refre,diff,dz,dx,velmin=-100,velmax=100,thresh=5,pclip=0.1,agc=False,show=False)

#ano = create_constptb_loc(nz,nx,ptb=0.95,naz=200,nax=550,cz=550,cx=375,rectx=100,rectz=100)

#ano1 = create_randomptb_loc(nz,nx,romin=0.95,romax=1.0,naz=220,nax=350,cz=525,cx=280,nptsx=2,octaves=2,period=80,
#                           persist=0.2,ncpu=1,sigma=20)
#ano2 = create_randomptb_loc(nz,nx,romin=0.95,romax=1.0,naz=150,nax=405,cz=575,cx=560,nptsx=2,octaves=2,period=80,
#                           persist=0.2,ncpu=1,sigma=20)

ano1 = create_randomptb_loc(nz,nx,romin=0.97,romax=1.0,naz=220,nax=200,cz=525,cx=550,nptsx=2,octaves=2,period=80,
                           persist=0.2,ncpu=1,sigma=20)

ano2 = create_randomptb_loc(nz,nx,romin=0.97,romax=1.0,naz=220,nax=100,cz=525,cx=200,nptsx=2,octaves=2,period=80,
                           persist=0.2,ncpu=1,sigma=20)

#ano = create_randomptbs_loc(nz,nx,nptbs=1,romin=0.8,romax=1.2,
#                            minnaz=100,maxnaz=250,minnax=500,maxnax=700,mincz=450,maxcz=500,mincx=350,maxcx=600,
#                            mindist=100,nptsz=2,nptsx=2,octaves=2,period=80,persist=0.2,ncpu=1,sigma=20)

#ano = ano2*ano1
velout2 = velw*ano1*ano2
ndiff = velout2 - vel

plot_imgvelptb(refre,ndiff*1000.0,dz,dx,velmin=-100,velmax=100,thresh=5,pclip=0.1,agc=False,show=False)
plt.show()

#plt.figure()
#plt.imshow(vel,cmap='jet',vmin=vmin,vmax=vmax)
#plt.figure()
#plt.imshow(ano,cmap='seismic')
#plt.figure()
#plt.imshow(velout2,cmap='jet',vmin=vmin,vmax=vmax)
#plt.show()

#sep.write_file("sigsbee_velsaltw2.H",velout ,ds=vaxes.d,os=vaxes.o)
sep.write_file("sigsbee_veloverw2.H",velout2,ds=vaxes.d,os=vaxes.o)


