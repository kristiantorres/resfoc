import inpout.seppy as seppy
import numpy as np
from resfoc.resmig import postresmig,convert2time,get_rho_axis
from scaas.gradtaper import build_taper, build_taper_ds, build_taper_bot
from scaas.off2ang import off2angkzx, get_angkzx_axis
from genutils.plot import plot_img2d
from genutils.movie import viewimgframeskey

sep = seppy.sep()

# Read in the defocused image
#iaxes,img = sep.read_file("spimgextbobdistrwrng.H")
iaxes,img = sep.read_file("resfaultfocuswindtstkro1.H")
img  = img.reshape(iaxes.n,order='F')
imgt = np.ascontiguousarray(img.T).astype('float32')
#imgtw = imgt[20,0,30:542,:450]
#imgtw = imgt[30:542,:450]
imgtw = imgt

# Get axes
nx,nz = imgtw.shape; oz,ox = iaxes.o; dz,dx = iaxes.d

# Taper the top of the image
_,tap = build_taper_bot(nx,nz,400,430)

timgtw = np.zeros(imgtw.shape,dtype='float32')
timgtw = tap.T*imgtw

sc = 0.5
imin = sc*np.min(timgtw); imax = sc*np.max(timgtw)

plot_img2d(timgtw.T,aspect='auto')

# Depth Residual migration
inro = 81; idro = 0.001250
rmig = postresmig(timgtw,[dx,dz],nps=[512,450],nro=inro,dro=idro,nthreads=1,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Convert to time
dtd = 0.004
rmigt = convert2time(rmig,dz,dt=dtd,dro=odro,verb=True)

viewimgframeskey(rmig,pclip=0.4,show=False)
viewimgframeskey(rmigt,pclip=0.4,show=True)

sep.write_file("postresfaultfocuswind.H",rmig.T  ,ds=[dz,dx,odro],os=[oz,7.37,ooro])
sep.write_file("postresfaultfocuswindt.H",rmigt.T,ds=[dz,dx,odro],os=[oz,7.37,ooro])

