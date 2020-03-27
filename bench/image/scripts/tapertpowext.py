import inpout.seppy as seppy
from scaas.gradtaper import build_taper
from resfoc.tpow import tpow
import matplotlib.pyplot as plt
from deeplearn.utils import normalize

sep = seppy.sep([])

iaxes,img = sep.read_file(None,ifname="fltimgext.H")

img = img.reshape(iaxes.n,order='F')

nz = iaxes.n[0]; nx = iaxes.n[1]
oz = iaxes.o[0]; dz = iaxes.d[0]

tap1d,tap = build_taper(nx,nz,20,100)
tapd = img*tap

plt.imshow(tapd,cmap='gray'); plt.show()

imgo = tpow(tapd,nz,oz,dz,nx,1.6)
plt.imshow(normalize(imgo),cmap='gray',vmin=-4,vmax=4); plt.show()

sep.write_file(None,iaxes,imgo,ofname='fltimgwrng3prc.H')
