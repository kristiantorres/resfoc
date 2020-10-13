import inpout.seppy as seppy
import numpy as np
from deeplearn.python_patch_extractor import PatchExtractor
from deeplearn.utils import plot_patchgrid2d
from genutils.plot import plot_img2d
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Read in the poorly focused image
daxes,dfc = sep.read_file("spimgbobangwrng.H")
dfc = dfc.reshape(daxes.n,order='F').T
# Window and stack
dfcw = dfc[:,0,20:,:]
dstk = np.sum(dfcw,axis=1)

# Read in the refocused image
raxes,res = sep.read_file("faultfocusang.H")
[nz,na,nx] = raxes.n; [oz,oa,ox] = raxes.o; [dz,da,dx] = raxes.d
res = res.reshape(raxes.n,order='F').T
rstk = np.sum(res,axis=1)

# Read in the well-focused image
faxes,foc = sep.read_file("spimgbobang.H")
foc = foc.reshape(faxes.n,order='F').T
# Window and stack
focw = foc[:,0,20:,:]
fstk = np.sum(focw,axis=1)

# First window defocused and focused
dstkw = dstk[20:580,:]
fstkw = fstk[20:580,:]

begz = 100; endz = 356
begx = 10;  endx = 522
# Window all to target region
dstkww = dstkw[begx:endx,begz:endz]
fstkww = fstkw[begx:endx,begz:endz]
rstkw  = rstk [begx:endx,begz:endz]

# Plot the target region
#plot_img2d(dstkww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,title='Defocused',pclip=0.5,aspect=2.0,show=False)
#plot_img2d(rstkw.T, ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,title='Refocused',pclip=0.5,aspect=2.0,show=False)
#plot_img2d(fstkww.T,ox=ox+begx*dx,dx=dx,oz=begz*dz,dz=dz,title='Focused'  ,pclip=0.5,aspect=2.0,show=True )
#viewimgframeskey([dstkw,rstk,fstkw],pclip=0.5)

# Patch each of them
nzp,nxp = 128,128
pe = PatchExtractor.PatchExtractor((nzp,nxp),stride=(nzp//2,nxp//2))
fptch = pe.extract(fstkww.T)

# Plot the patch grids
plot_patchgrid2d(fstkww.T,nzp,nxp,dx=dx,dz=dz,pclip=0.5)

# Plot some patches of interest

#plot_img2d(fptch[1,3],dx=dx,dz=dz,show=True,aspect=2.0)

for i in range(fptch.shape[0]):
  for j in range(fptch.shape[1]):
    plot_img2d(fptch[i,j],dx=dx,dz=dz,show=True,aspect=2.0,title='%d,%d'%(i,j))

