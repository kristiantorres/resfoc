import inpout.seppy as seppy
import numpy as np
import velocity.mdlbuild as mdlbuild
from genutils.rand import randfloat
from genutils.plot import plot_img2d, plot_vel2d
from deeplearn.utils import plot_seglabel
import matplotlib.pyplot as plt

sep = seppy.sep()
iaxes,img = sep.read_file("faultfocusang.H")
[nz,na,nx] = iaxes.n; [oz,oa,ox] = iaxes.o; [dz,da,dx] = iaxes.d
img = img.reshape(iaxes.n,order='F').T
stk = np.sum(img,axis=1)
stk3d = np.repeat(stk[np.newaxis],20,axis=0)

dzm = dz*1000; dxm = dx*1000
oxm = ox*1000

# Build a model that is the same size
minvel = 1600; maxvel = 5000
nlayer = 200
mb = mdlbuild.mdlbuild(nx,dxm,20,dy=dxm,dz=dzm,basevel=5000)
props = mb.vofz(nlayer,minvel,maxvel,npts=2)
thicks = np.random.randint(5,15,nlayer)

# Randomize the squishing depth
# sqmin = 180; sqmax = 198
sqz = np.random.choice(list(range(180,199)))

dlyr = 0.05
for ilyr in range(nlayer):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,
             layer=50,layer_rand=0.00,dev_layer=dlyr)
  if(ilyr == sqz):
    mb.squish(amp=150,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=3,order=3)

mb.deposit(1480,thick=40,layer=150,dev_layer=0.0)
mb.trim(top=0,bot=900)

# Pos
xpos = np.asarray([0.18,0.26,0.44,0.60,0.68,0.77])
xhi = xpos + 0.04
xlo = xpos - 0.04
cxpos = np.zeros(xpos.shape)

rd = 20
nflt = len(xpos)
for iflt in range(nflt):
  cxpos[iflt] = randfloat(xlo[iflt],xhi[iflt])
  if(iflt > 0 and cxpos[iflt] - cxpos[iflt-1] < 0.04):
    cxpos[iflt] += 0.04
  cdaz = randfloat(16000,20000)
  cdz = cdaz + randfloat(0,6000)
  # Choose the theta_die
  theta_die = randfloat(1.5,3.5)
  if(theta_die < 2.7):
    begz = randfloat(0.23,0.26)
  else:
    begz = randfloat(0.26,0.33)
  print(cdaz,cdz,begz,theta_die)
  fpr = np.random.choice([True,True,False])
  mb.fault2d(begx=cxpos[iflt],begz=begz,daz=cdaz,dz=cdz,azim=180,theta_die=theta_die,theta_shift=4.0,dist_die=2.0,
             throwsc=35.0,fpr=fpr,rectdecay=rd)
velw = mb.vel
refw = mb.get_refl2d()
lblw = mb.get_label2d()

# First window defocused and focused
velw = velw[20:580,:]
refw = refw[20:580,:]
stkw = stk [20:580,:]
lblw = lblw[20:580,:]

begz = 100; endz = 356
begx = 10;  endx = 522
# Window all to target region
velww = velw[begx:endx,begz:endz]
refww = refw[begx:endx,begz:endz]
lblww = lblw[begx:endx,begz:endz]
stkww = stkw[begx:endx,begz:endz]

plot_img2d(stkw.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_vel2d(velw.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_img2d(refw.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_seglabel(refw.T,lblw.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=True)

