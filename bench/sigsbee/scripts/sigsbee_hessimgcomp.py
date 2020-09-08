import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2angkzx, get_angkzx_axis
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the extended image
iaxes,img = sep.read_file("sigextimg.H")
img  = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
imgn = img[np.newaxis]
[nz,nx,ny,nhx] = iaxes.n; [oz,ox,oy,ohx] = iaxes.o; [dz,dx,dy,dhx] = iaxes.d
sc = 0.1
imin = sc*np.min(img); imax = sc*np.max(img)

# Read in the Hessian
haxes,hes = sep.read_file("sigsbee_hess_test.H")
hes  = np.ascontiguousarray(hes.reshape(haxes.n,order='F').T).astype('float32')
hesn = hes[np.newaxis]
hmin = sc*np.min(hes); hmax = sc*np.max(hes)

# Convert to angle
na = 64
iang = off2angkzx(imgn,ohx,dhx,dz,na=na,nthrds=10,transp=True,cverb=False)
hang = off2angkzx(hesn,ohx,dhx,dz,na=na,nthrds=10,transp=True,cverb=False)
na,oa,da = get_angkzx_axis(na,amax=60)

sep.write_file("sigimgang.H",iang.T,os=[oz,oa,0.0,ox,0.0],ds=[dz,da,1.0,dx,1.0])
sep.write_file("sighesang.H",hang.T,os=[oz,oa,0.0,ox,0.0],ds=[dz,da,1.0,dx,1.0])

plt.figure()
plt.imshow(img[20,0,:,:].T,cmap='gray',interpolation='bilinear',aspect='auto',vmin=imin,vmax=imax)
plt.figure()
plt.imshow(hes[20,0,:,:].T,cmap='gray',interpolation='bilinear',aspect='auto',vmin=hmin,vmax=hmax)
plt.show()

