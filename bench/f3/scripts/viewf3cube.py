import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewcube3d

sep = seppy.sep()

caxes1,cub1 = sep.read_file("f3tmps60hz/f3img60hz-12.H")
#caxes2,cub2 = sep.read_file("f3tmps700/f3img700-01.H")
caxes2,cub2 = sep.read_file("f3tmpsgncw/f3imggncw-12.H")
dz,dx,dy = caxes1.d; oz,ox,oy = caxes1.o
cub1 = np.ascontiguousarray(cub1.reshape(caxes1.n,order='F').T).astype('float32')
cub2 = np.ascontiguousarray(cub2.reshape(caxes2.n,order='F').T).astype('float32')
dz,dx,dy = caxes1.d; oz,ox,oy = caxes1.o
gcub1 = agc(cub1)
gcub2 = agc(cub2)

pclip = 0.1
imin = pclip*np.min(gcub1)
imax = pclip*np.max(gcub2)

# Taper the top of the image
viewcube3d(gcub1[:,:500,100:300].T,ds=[dz,dx,dy],os=[oz,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',pclip=0.05,width3=1.0,vmin=imin,vmax=imax,show=False)

viewcube3d(gcub2[:,:500,100:300].T,ds=[dz,dx,dy],os=[oz,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',pclip=0.05,width3=1.0,vmin=imin,vmax=imax,show=True)

