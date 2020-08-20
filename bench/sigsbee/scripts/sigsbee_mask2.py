import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in velocity model
vaxes,vel = sep.read_file("sigvelsalt2_interp.H")
vel = vel.reshape(vaxes.n,order='F')
salt = 4.5

idx = vel >= salt
msk = np.ascontiguousarray(np.copy(vel)).astype('float32')
msk[idx] = 0.0
msk[~idx] = 1.0
mskw = msk[:,0,:]

plt.imshow(mskw.T,cmap='gray')
plt.show()

smmsk = smooth(mskw,rect1=30,rect2=30)
idx2 = smmsk > 0.95
smmsk[idx2] = 1.0
smmsk[~idx2] = 0.0
smmsk2 = smooth(smmsk,rect1=2,rect2=2)

# Read in images
iaxes,img = sep.read_file("sigsaltw2.H")
[nz,nx,ny,nhx] = iaxes.n; [oz,ox,oy,ohx] = iaxes.o; [dz,dx,dy,dhx] = iaxes.d
img = img.reshape(iaxes.n,order='F').T
imin = np.min(img); imax = np.max(img)
scale = 0.1

msked = np.zeros(img.shape,dtype='float32')

for ihx in range(nhx):
  msked[ihx]  = smmsk2*img[ihx]

sep.write_file('sigsaltw2masked.H',msked.T,ds=iaxes.d,os=iaxes.o)

