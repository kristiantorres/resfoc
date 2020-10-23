import inpout.seppy as seppy
import numpy as np
import velocity.mdlbuild as mdlbuild
from genutils.plot import plot_img2d, plot_vel2d
from deeplearn.utils import plot_seglabel
import matplotlib.pyplot as plt

sep = seppy.sep()
iaxes,img = sep.read_file("faultfocusang.H")
[nz,na,nx] = iaxes.n; [oz,oa,ox] = iaxes.o; [dz,da,dx] = iaxes.d
img = img.reshape(iaxes.n,order='F').T
stk = np.sum(img,axis=1)
stk3d = np.repeat(stk[np.newaxis],20,axis=0)

ox *= 1000; dx *= 1000
dz *= 1000

# Build a model that is the same size
minvel = 1600; maxvel = 5000
nlayer = 200
mb = mdlbuild.mdlbuild(nx,dx,20,dy=dx,dz=dz,basevel=5000)
props = mb.vofz(nlayer,minvel,maxvel,npts=2)
thicks = np.random.randint(5,15,nlayer)

dlyr = 0.05
for ilyr in range(nlayer):
  mb.deposit(velval=props[ilyr],thick=thicks[ilyr],dev_pos=0.0,
             layer=50,layer_rand=0.00,dev_layer=dlyr)

mb.trim(top=0,bot=900)

fpr = False
mb.vel = stk3d
#mb.fault2d(begx=0.26,begz=0.4,daz=8000,dz=12000,azim=180.0,theta_die=11,theta_shift=4.0,dist_die=2.0,
#           throwsc=35.0,fpr=fpr,thresh=0.15)
#mb.fault2d(begx=0.25,begz=0.2,daz=16000,dz=20000,azim=180.0,theta_die=2,theta_shift=4.0,dist_die=2.0,
#           throwsc=35.0,fpr=fpr)
#mb.fault2d(begx=0.545,begz=0.4,daz=8000,dz=12000,azim=180.0,theta_die=11,theta_shift=4.0,dist_die=2.0,
#           throwsc=35.0,fpr=fpr)
#mb.fault2d(begx=0.66,begz=0.4,daz=8000,dz=9000,azim=180.0,theta_die=11,theta_shift=4.0,dist_die=2.0,
#           throwsc=30.0,fpr=fpr)
#mb.fault2d(begx=0.76,begz=0.4,daz=8000,dz=9000,azim=180.0,theta_die=11,theta_shift=4.0,dist_die=2.0,
#           throwsc=35.0,fpr=fpr)
mb.fault2d(begx=0.77,begz=0.26,daz=20000,dz=24000,azim=180.0,theta_die=2.5,theta_shift=4.0,dist_die=2.0,
           throwsc=35.0,fpr=fpr)
velw = mb.vel
refw = mb.get_refl2d()
lblw = mb.get_label2d()

#plt.imshow(lblw.T); plt.show()

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

plot_img2d(stkww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_img2d(velww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_img2d(refww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=False)
plot_seglabel(refww.T,lblww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,pclip=0.5,aspect=2.0,show=True)

