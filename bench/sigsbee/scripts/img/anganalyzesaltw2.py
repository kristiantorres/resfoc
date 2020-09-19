import inpout.seppy as seppy
import numpy as np
from utils.movie import viewcube3d

sep = seppy.sep()

iaxes,img = sep.read_file("resmsksaltw2t.H")
[oz,oa,ox,oro] = iaxes.o; [dz,da,dx,dro] = iaxes.d

img = img.reshape(iaxes.n,order='F').T

imgw = img[20,:,32:,:]
imgwt = np.transpose(imgw,(1,0,2))

viewcube3d(imgwt,ds=[dz,da,dx],os=[oz,0,ox],
           interp='none',cmap='gray',pclip=0.2,transp=True,width3=1.)


