import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewcube3d

sep = seppy.sep()

caxes,cub = sep.read_file("f3tmps2/f3img2-12.H")
dz,dx,dy = caxes.d; oz,ox,oy = caxes.o
cub = np.ascontiguousarray(cub.reshape(caxes.n,order='F').T).astype('float32')
dz,dx,dy = caxes.d; oz,ox,oy = caxes.o
gcub = agc(cub)

# Taper the top of the image
viewcube3d(gcub.T,ds=[dz,dx,dy],os=[oz,ox,oy],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',pclip=0.1,width3=1.0)

