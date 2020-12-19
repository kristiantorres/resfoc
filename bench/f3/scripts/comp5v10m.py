import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewcube3d

sep = seppy.sep()

# Read in 10m image
caxes1,cub1 = sep.read_file("f3imgs/f3tmpsgncw/f3imggncw-12.H")
dz1,dx1,dy1 = caxes1.d; oz1,ox1,oy1 = caxes1.o
cub1 = np.ascontiguousarray(cub1.reshape(caxes1.n,order='F').T).astype('float32')
gcub1 = agc(cub1)

# Read in 5m image
caxes2,cub2 = sep.read_file("f3imgs/f3img5m/f3img5m-12.H")
dz2,dx2,dy2 = caxes2.d; oz2,ox2,oy2 = caxes2.o
cub2 = np.ascontiguousarray(cub2.reshape(caxes2.n,order='F').T).astype('float32')
gcub2 = agc(cub2)

pclip = 0.2
imin = pclip*np.min(gcub2)
imax = pclip*np.max(gcub2)

#viewcube3d(gcub1[20:-20,20:-20,100:300].T,ds=[dz1,dx1,dy1],os=[250*dz1,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
#           interp='none',width3=1.0,pclip=0.3,vmin=imin,vmax=imax,cmap='gray',show=False)

viewcube3d(gcub2[20:-20,20:-20,100:700].T,ds=[dz2,dx2,dy2],os=[100*dz2,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',width3=1.0,pclip=0.2,vmin=imin,vmax=imax,cmap='gray',show=True)

