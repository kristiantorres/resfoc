import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import resample
import matplotlib.pyplot as plt

sep = seppy.sep()

saxes,sig = sep.read_file("hale_trdata.H")
[nt,ntr] = saxes.n
sig = sig.reshape(saxes.n,order='F').T

naxes,noz = sep.read_file("hale_noise.H")
noz = noz.reshape(naxes.n,order='F').T

daxes,dat = sep.read_file("hale_shotflatbob.H")
dat = dat.reshape(daxes.n,order='F').T
sc = 0.4
dmin = sc*np.min(dat); dmax = sc*np.max(dat)

# Stretch the noise
nozstr = resample(noz,[ntr,nt],kind='quintic')
print(nozstr.shape)

# Normalize the noise
nozstr = (nozstr/np.max(nozstr))*0.2

sigpnoz = sig + nozstr
vmin = sc*np.min(sigpnoz); vmax = sc*np.max(sigpnoz)

plt.figure()
plt.imshow(sigpnoz[300:600,:1000].T,cmap='gray',aspect='auto',vmin=vmin,vmax=vmax)
plt.figure()
plt.imshow(dat[300:600,:].T,cmap='gray',aspect='auto',vmin=dmin,vmax=dmax)

plt.show()
