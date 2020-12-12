import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewcube3d

sep = seppy.sep()

caxes1,cub1 = sep.read_file("f3img700a-12.H")
caxes2,cub2 = sep.read_file("f3sos.H")
dz,dx,dy = caxes1.d; oz,ox,oy = caxes1.o
cub1 = np.ascontiguousarray(cub1.reshape(caxes1.n,order='F').T).astype('float32')
cub2 = np.ascontiguousarray(cub2.reshape(caxes2.n,order='F').T).astype('float32')
dz,dx,dy = caxes1.d; oz,ox,oy = caxes1.o
gcub1 = agc(cub1,rect1=250)
gcub2 = agc(cub2,rect1=250)

pclip = 0.05
imin = pclip*np.min(gcub1)
imax = pclip*np.max(gcub2)

# Taper the top of the image
viewcube3d(gcub1[:,:,100:400].T,ds=[dz,dx,dy],os=[oz,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',pclip=0.05,width3=1.0,vmin=imin,vmax=imax,show=False)

viewcube3d(gcub2[:,:,100:400].T,ds=[dz,dx,dy],os=[oz,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',pclip=0.05,width3=1.0,vmin=imin,vmax=imax,show=True)

