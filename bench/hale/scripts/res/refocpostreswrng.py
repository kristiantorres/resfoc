import inpout.seppy as seppy
import numpy as np
from scaas.off2ang import off2angkzx
from resfoc.resmig import postresmig, get_rho_axis, convert2time
from resfoc.estro import refocusimg
from genutils.plot import plot_img2d

sep = seppy.sep()
# Read in the image
iaxes,img = sep.read_file("spimgbobangwrng.H")
img  = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
dz,da,daz,dx = iaxes.d; oz,oa,oaz,ox = iaxes.o
imgw = img[30:542,0,:,:450]
stk = np.sum(imgw,axis=1)

# Read in Rho
raxes,rho = sep.read_file("faultfocusrhowind.H")
rho = np.ascontiguousarray(rho.reshape(raxes.n,order='F').T)

# Residual migration
inro = 81; idro = 0.001250
pst = postresmig(stk,[dx,dz],nps=[512,512],nro=inro,dro=idro,nthreads=1,verb=True)
onro,ooro,odro = get_rho_axis(nro=inro,dro=idro)

# Convert to time
dtd = 0.004
rmigt = convert2time(pst,dz,dt=dtd,dro=odro,verb=True)

rfi = refocusimg(rmigt,rho,odro)

plot_img2d(rfi[:,100:356].T,show=False)
plot_img2d(stk[:,100:356].T)

sep.write_file("faultfocusrfiwindpost.H",rfi[:,100:356].T,os=[oz,ox])

