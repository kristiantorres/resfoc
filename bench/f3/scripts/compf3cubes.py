import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewcube3d

sep = seppy.sep()

# Read in my image
caxes,cub = sep.read_file("f3extimgs/f3ang_tot.H")
dz,da,daz,dx,dy = caxes.d; oz,oa,oaz,ox,oy = caxes.o
cub = np.ascontiguousarray(cub.reshape(caxes.n,order='F').T).astype('float32')
cubt = np.transpose(cub,(2,3,0,1,4)) #[ny,nx,naz,na,nz] -> [naz,na,ny,nx,nz]
stk = np.sum(cub,axis=3)[:,:,0,:]
gstk = agc(stk)

# Read in their image
faxes,f3 = sep.read_file("migwt.T")
f3 = f3.reshape(faxes.n,order='F').T
dt = faxes.d[-1]

pclip = 0.3
imin = pclip*np.min(gstk)
imax = pclip*np.max(gstk)

viewcube3d(gstk[20:-20,20:-20,200:600].T,ds=[dz,dx,dy],os=[200*dz,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',width3=1.0,vmin=imin,vmax=imax,cmap='gray',show=False)

viewcube3d(f3[250:650,20:480,45:105],ds=[dt,dx,dy],os=[200*dt,0,0],label3='Time (s)',label1='X (km)',label2='Y (km)',
           interp='bilinear',width3=1.0,cmap='gray',show=True)

