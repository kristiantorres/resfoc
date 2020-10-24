import inpout.seppy as seppy
import numpy as np
import velocity.mdlbuild as mdlbuild
from genutils.plot import plot_img2d, plot_vel2d
from deeplearn.utils import plot_seglabel
import matplotlib.pyplot as plt

sep = seppy.sep()
iaxes,img = sep.read_file("faultfocusang.H")
[nz,na,nx] = iaxes.n; [oz,oa,ox] = iaxes.o; [dz,da,dx] = iaxes.d
nx = 800
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

dlyr = 0.05
for ilyr in range(nlayer):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,
             layer=50,layer_rand=0.00,dev_layer=dlyr)
  if(ilyr == 195):
    mb.squish(amp=150,azim=90.0,lam=0.4,rinline=0.0,rxline=0.0,mode='perlin',octaves=3,order=3)

mb.deposit(1480,thick=40,layer=150,dev_layer=0.0)
mb.trim(top=0,bot=900)

fpr = True; rd = 20
#mb.vel = stk3d
mb.fault2d(begx=0.25,begz=0.28,daz=19000,dz=24000,azim=180.0,theta_die=3.0,theta_shift=4.0,dist_die=2.0,
           throwsc=35.0,fpr=fpr,rectdecay=rd)
mb.fault2d(begx=0.30,begz=0.21,daz=8800,dz=12000,azim=180.0,theta_die=4.4,theta_shift=4.0,dist_die=2.0,
           throwsc=35.0,fpr=fpr,rectdecay=rd)
mb.fault2d(begx=0.432,begz=0.25,daz=20000,dz=26000,azim=180.0,theta_die=2.5,theta_shift=4.0,dist_die=2.0,
           throwsc=35.0,fpr=fpr,rectdecay=rd)
mb.fault2d(begx=0.544,begz=0.3,daz=16000,dz=18000,azim=180.0,theta_die=4.0,theta_shift=4.0,dist_die=2.0,
           throwsc=30.0,fpr=fpr,rectdecay=rd)
mb.fault2d(begx=0.6,begz=0.26,daz=18000,dz=18000,azim=180.0,theta_die=2.5,theta_shift=4.0,dist_die=2.0,
           throwsc=35.0,fpr=fpr,rectdecay=rd)
mb.fault2d(begx=0.663,begz=0.26,daz=20000,dz=24000,azim=180.0,theta_die=2.5,theta_shift=4.0,dist_die=2.0,
           throwsc=35.0,fpr=fpr,rectdecay=rd)
velw = mb.vel
refw = mb.get_refl2d()
lblw = mb.get_label2d()

# First window defocused and focused
velw = velw[120:660,:]
refw = refw[120:660,:]
stkw = stk [20:580,:]
lblw = lblw[120:660,:]

#print(refw.shape,stkw.shape)

begz = 100; endz = 356
begx = 10;  endx = 522
# Window all to target region
velww = velw[begx:endx,begz:endz]
refww = refw[begx:endx,begz:endz]
lblww = lblw[begx:endx,begz:endz]
stkww = stkw[begx:endx,begz:endz]

plot_img2d(stkww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_vel2d(velww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_img2d(refww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_seglabel(stkww.T,lblww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=True)

