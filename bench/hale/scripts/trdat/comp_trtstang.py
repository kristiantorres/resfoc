import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import normextract
from genutils.plot import plot_img2d, plot_cubeiso
from genutils.movie import viewcube3d

sep = seppy.sep()

# Read in training image
iaxes,imgs = sep.read_wind("hale_deftrimgscln.H",fw=10,nw=1)
imgs = imgs.reshape(iaxes.n,order='F').T
imgw = imgs[:,0,:,:]

# Read in real image
raxes,rel = sep.read_file("spimgbobangwrng.H")
oz,oa,oy,ox = raxes.o; dz,da,dy,dx = raxes.d
rel = rel.reshape(raxes.n,order='F').T

imgww = imgw[100:356,31:,140:652]
imgww = imgw[140:652,31:,100:356]
relw  = rel[30:542,0,31:,100:356]

imgwstk = agc(np.sum(imgww,axis=1))
relwstk = agc(np.sum(relw,axis=1))

#plot_img2d(imgwstk.T,pclip=0.5,show=False)
#plot_img2d(relwstk.T,pclip=0.5)

imgwwt = np.transpose(imgww,(2,0,1))
relwt  = np.transpose(relw ,(2,0,1))
viewcube3d(imgwwt,ds=[dz,da,dx],show=False,width3=1.0)
viewcube3d(relwt,ds=[dz,da,dx],show=True,width3=1.0)

