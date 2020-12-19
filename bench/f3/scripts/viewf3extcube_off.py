import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from genutils.movie import viewcube3d

sep = seppy.sep()

caxes,cub = sep.read_file("f3extimgs/f3extimgio5m_tot.H")
dz,dx,dy,dhx = caxes.d; oz,ox,oy,ohx = caxes.o
cub = np.ascontiguousarray(cub.reshape(caxes.n,order='F').T).astype('float32')
gcub = agc(cub[20])

pclip = 0.2
imin = pclip*np.min(gcub)
imax = pclip*np.max(gcub)

# Taper the top of the image
viewcube3d(gcub[:,20:-20,100:700].T,ds=[dz,dx,dy],os=[oz,0,0],label3='Z (km)',label1='X (km)',label2='Y (km)',
           interp='bilinear',width3=1.0,vmin=imin,vmax=imax,show=False)

print(ohx)
viewcube3d(cub[:,40,20:-20,100:700].T,ds=[dz,dhx,dx],os=[oz,ohx,ox],label3='Z (km)',label1='X (km)',label2='hx (km)',
           interp='bilinear',width3=1.0,pclip=0.1,show=True)

