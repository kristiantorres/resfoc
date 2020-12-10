import inpout.seppy as seppy
import numpy as np
from oway.utils import interp_vel
from genutils.plot import plot_img2d, plot_vel2d, plot_imgpang, plot_rhoimg2d
from genutils.ptyprint import create_inttag
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the velocity model
vaxes,vel = sep.read_file("velfltfocus1.H")
vel = vel.reshape(vaxes.n,order='F')
nz,nvx = vaxes.n; dz,dvx = vaxes.d; oz,ovx = vaxes.o

# Read in the poorly-focused image
sep = seppy.sep()
iaxes,img = sep.read_file("spimgbobangwrng.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,0,:,:]
#stkw = agc(np.sum(img,axis=1))[30:542,:512]
stkw = np.sum(img,axis=1)[30:542,:512]
dz,da,dy,dx = iaxes.d; oz,oa,oy,ox = iaxes.o
nx,nz = stkw.shape

# Interpolate the velocity model
nxr = iaxes.n[-1]
velin = np.zeros([nz,1,nvx],dtype='float32')
velin[:,0,:] = vel[:nz,:]
veli = interp_vel(nz,1,0.0,1.0,nxr,ox,dx,
                  velin,dvx,1.0,ovx=ovx,ovy=0.0)

# Plotting origins
oxp,ozp = ox + 30*dx, 100*dz

plot_vel2d(veli[:,0,:nx],dx=dx,dz=dz,ox=oxp,aspect=4.0,barx=0.7,
           figname='./fig/fall2020/vel.png')

imin,imax = np.min(stkw),np.max(stkw)
plot_img2d(stkw.T,dx=dx,dz=dz,ox=oxp,aspect=3.0,ox_box=oxp,oz_box=ozp,nx_box=nx,nz_box=256,
           imin=imin,imax=imax,pclip=0.5,figname='./fig/fall2020/imgbox.png')

# Plot an angle gather in that region
imgw = img[30:542,32:,100:356]
imgwt = np.transpose(imgw,(1,2,0))

#print(ox + (30+225)*dx)
plot_imgpang(imgwt,dx,dz,225,0.0,da,ox=oxp,oz=ozp,ipclip=0.5,apclip=0.5,iaspect=3.0,aaspect=250,wspace=0.1,
             wratio=12,figname='./fig/fall2020/angdefoc.png')

# Residual migration images
raxes,res = sep.read_file("resfaultfocuswindtstk.H")
res = res.reshape(raxes.n,order='F').T
nro = raxes.n[-1]; dro = raxes.d[-1]; oro = raxes.o[-1]

#for iro in range(nro):
#  tag = create_inttag(iro,nro)
#  fname = './fig/fall2020/ro-%s.png'%(tag)
#  ro = oro + iro*dro
#  plot_img2d(res[iro,:,100:356].T,dx=dx,dz=dz,ox=oxp,oz=ozp,figname=fname,aspect=3.0,
#             imin=imin,imax=imax,pclip=0.5,title=r'$\rho$=%.4f'%(ro))

# Read in rho
waxes,rho = sep.read_file("faultfocusrhowind.H")
rho = rho.reshape(waxes.n,order='F').T

# Plot picked rho on top of image
plot_rhoimg2d(stkw[:,100:356].T,rho.T,aspect=3.0,ox=oxp,oz=ozp,dx=dx,dz=dz,imin=imin,imax=imax,pclip=0.5,
              figname='./fig/fall2020/sembrho.png')

# Read in the refocused image
waxes,rfi = sep.read_file("faultfocusrfiwind.H")
rfi = rfi.reshape(waxes.n,order='F').T

plot_img2d(rfi.T,dx=dx,dz=dz,ox=oxp,oz=ozp,aspect=3.0,
           imin=imin,imax=imax,pclip=0.5,figname='./fig/fall2020/sembrfi.png')

# Read in an angle gather image
faaxes,fang = sep.read_file("./dat/split_angs/fimgn-00222.H")
fang = fang.reshape(faaxes.n,order='F')[:,:,32:].T
fangt = np.transpose(fang,(0,2,1))

plot_imgpang(fangt,dx,dz,225,0.0,da,ox=oxp,oz=ozp,ipclip=0.5,apclip=0.5,iaspect=3.0,aaspect=250,wspace=0.1,
             wratio=12,figname='./fig/fall2020/trnangfoc.png')

fsaaxes,fsang = sep.read_file("./dat/split_angs/fimgn-00222-sos.H")
fsang = fsang.reshape(fsaaxes.n,order='F')[:,:,32:].T
fsangt = np.transpose(fsang,(0,2,1))
plot_imgpang(fsangt,dx,dz,225,0.0,da,ox=oxp,oz=ozp,ipclip=0.5,apclip=0.5,iaspect=3.0,aaspect=250,wspace=0.1,
             wratio=12,figname='./fig/fall2020/trnangfocsos.png')

daaxes,dang = sep.read_file("./dat/split_angs/dimgn-00222.H")
dang = dang.reshape(daaxes.n,order='F')[:,:,32:].T
dangt = np.transpose(dang,(0,2,1))
plot_imgpang(dangt,dx,dz,225,0.0,da,ox=oxp,oz=ozp,ipclip=0.5,apclip=0.5,iaspect=3.0,aaspect=250,wspace=0.1,
             wratio=12,figname='./fig/fall2020/trnangdef.png')

dsaaxes,dsang = sep.read_file("./dat/split_angs/dimgg-00222-sos.H")
dsang = dsang.reshape(dsaaxes.n,order='F')[:,:,32:].T
dsangt = np.transpose(dsang,(0,2,1))
plot_imgpang(dsangt,dx,dz,225,0.0,da,ox=oxp,oz=ozp,ipclip=0.5,apclip=0.5,iaspect=3.0,aaspect=250,wspace=0.1,
             wratio=12,figname='./fig/fall2020/trnangdefsos.png')

# Read in rho predicted from DNN
waxes,dnnrho = sep.read_file("realtorch_rho.H")
dnnrho = dnnrho.reshape(waxes.n,order='F').T
plot_rhoimg2d(stkw[:,100:356].T,dnnrho.T,aspect=3.0,ox=oxp,oz=ozp,dx=dx,dz=dz,imin=imin,imax=imax,pclip=0.5,
              figname='./fig/fall2020/dnnrho.png')

waxes,dnnrfi = sep.read_file("realtorch_rfi.H")
dnnrfi = dnnrfi.reshape(waxes.n,order='F').T
plot_img2d(dnnrfi.T,dx=dx,dz=dz,ox=oxp,oz=ozp,aspect=3.0,
           imin=imin,imax=imax,pclip=0.5,figname='./fig/fall2020/dnnrfi.png')

