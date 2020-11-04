import inpout.seppy as seppy
import numpy as np
from genutils.plot import plot_img2d

sep = seppy.sep()

# Read in training image
iaxes,imgs = sep.read_file("hale_foctrstks.H")
imgs = imgs.reshape(iaxes.n,order='F').T
imgw = imgs[0,:,0,:].T

# Read in real image
raxes,rel = sep.read_file("spimgbobangstkwind.H")
oz,ox = raxes.o; dz,dx = raxes.d
rel = rel.reshape(raxes.n,order='F')

# Read in smoothed image
saxes,smt = sep.read_file("sos.H")
smt = smt.reshape(saxes.n,order='F')

imgww = imgw[100:356,140:652]
relw = rel[100:356,30:542]
smtw = smt[100:356,30:542]

print(imgww.shape,relw.shape,smtw.shape)

plot_img2d(imgww,ox=ox+30*dx,oz=100*dz,dz=dz,dx=dx,show=False,pclip=0.5,aspect=2.0)
plot_img2d(relw,ox=ox+30*dx,oz=100*dz,dz=dz,dx=dx,show=False,pclip=0.4,aspect=2.0)
plot_img2d(smtw,ox=ox+30*dx,oz=100*dz,dz=dz,dx=dx,show=True,pclip=0.4,aspect=2.0)
