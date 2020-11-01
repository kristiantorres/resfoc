import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import plot_seglabel, normextract
from genutils.plot import plot_img2d
import h5py

sep = seppy.sep()

# Read in training data
hf = h5py.File('/net/thing/scr2/joseph29/hale2_fltseg128.h5','r')
keys = list(hf.keys())
ntr = len(keys)//2

# Read in test data
tsaxes,tst = sep.read_file("sos.H")
tst = tst.reshape(tsaxes.n,order='F')
tstw = tst[100:356,20:532]
plot_img2d(tstw)
tptch = normextract(tstw,nzp=128,nxp=128)
print(tptch.shape)

for itr in range(ntr):
  img = hf[keys[itr    ]][0,0]
  lbl = hf[keys[itr+ntr]][0,0]
  plot_img2d(img,show=False)
  plot_img2d(tptch[np.random.randint(0,21)])
  plot_seglabel(img,lbl,show=True)

