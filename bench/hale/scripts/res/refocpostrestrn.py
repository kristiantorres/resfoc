import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2angkzx
from resfoc.resmig import postresmig, get_rho_axis, convert2time
from resfoc.estro import refocusimg
from genutils.plot import plot_img2d

sep = seppy.sep()
# Read in the image
iaxes,img = sep.read_wind("hale_deftrimgs_off.H",fw=2,nw=1)
dz,dx,dy,dhx,dm = iaxes.d; oz,ox,oy,ohx,om = iaxes.o
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)[:,:,140:652,:450].astype('float32')
imgw = img[:,0,:,:]
zof = img[20,0,:,:]

# Read in Rho
raxes,rho = sep.read_file("resmigtrnrho.H")
rho = np.ascontiguousarray(rho.reshape(raxes.n,order='F').T)

# Convert to angle
na = 64
ang = off2angkzx(img[np.newaxis],ohx,dhx,dz,na=na,nthrds=20,transp=True)[0,:,0,:,:]
stk = np.sum(ang,axis=1)

print(stk.shape,rho.shape)

# Residual migration
inro = 81; idro = 0.001250
pst = postresmig(stk,[dx,dz],nps=[512,512],nro=inro,dro=idro,nthreads=1,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

dtd = 0.004
rmigt = convert2time(pst,dz,dt=dtd,dro=odro,verb=True)

rfi = refocusimg(rmigt,rho,odro)

sep.write_file("posresmigtrnrfi.H",rfi[:,100:356].T,os=[0,7.37],ds=[dz,dx])

#plot_img2d(rfi.T,show=False)
#plot_img2d(stk.T)

