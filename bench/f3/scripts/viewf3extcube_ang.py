import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc, tpow
from genutils.plot import plot_allanggats
from genutils.movie import viewcube3d

sep = seppy.sep()

caxes,cub = sep.read_file("f3extimgs/f3ang_tot.H")
dz,da,daz,dx,dy = caxes.d; oz,oa,oaz,ox,oy = caxes.o
cub = np.ascontiguousarray(cub.reshape(caxes.n,order='F').T).astype('float32')
cubt = np.transpose(cub,(2,3,0,1,4)) #[ny,nx,naz,na,nz] -> [naz,na,ny,nx,nz]
stk = np.sum(cub,axis=3)[:,:,0,:]
gstk = agc(stk)

pclip = 0.3
imin = pclip*np.min(gstk)
imax = pclip*np.max(gstk)

viewcube3d(gstk[20:-20,20:-20,100:700].T,ds=[dz,dx,dy],os=[dz*100,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',width3=1.0,vmin=imin,vmax=imax,show=False)

xline = 40
print("Extracting at y=%f"%((xline-20)*dy))
cubtw = cubt[0,20:,xline,20:-20,100:700]
gcubtw = tpow(cubtw,dt=dz,tpow=2,transp=True)
viewcube3d(gcubtw.T,ds=[dz,da,dx],os=[dz*100,0.0,0.0],label3='Z (km)',label1='X (km)',label2=r'Angle ($\degree$)',
           interp='bilinear',width3=1.0,pclip=0.3,show=False)

#TODO: using viewcube3d make it so you can visualize them in 3D by flattening them out
# View the gathers spatially
gcubtwt = np.transpose(gcubtw,(1,0,2))
plot_allanggats(gcubtwt,dz,dx,jx=10,zmin=dz*100,aagc=False,show=True,aspect='auto',interp='none',pclip=0.3)

