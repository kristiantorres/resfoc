import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import plot_seglabel, normextract
from genutils.plot import plot_img2d, plot_cubeiso
from genutils.movie import viewcube3d
import subprocess
from resfoc.gain import agc
import h5py

sep = seppy.sep()

# Read in training data
hf = h5py.File('/net/thing/scr2/joseph29/halefoc_sm-small.h5','r')
keys = list(hf.keys())
ntr = len(keys)//2

# Read in test data
tsaxes,tst = sep.read_file("spimgbobangwrng.H")
tst = tst.reshape(tsaxes.n,order='F').T
tstg = agc(tst[:,0,:,:])
tstw = tstg[20:532,32:,100:356]
tstwt = np.transpose(tstw,(1,0,2))
#viewcube3d(tstwt.T,width3=1.0)
iimg = "presmoothang.H"
oimg = "smoothang.H"
sep.write_file(iimg,tstwt.T)

# Apply SOS to the file
#pyexe  = '/sep/joseph29/anaconda3/envs/py37/bin/python'
#sosexe = '/homes/sep/joseph29/projects/resfoc/bench/hale/scripts/SOSmoothing.py'
#sp = subprocess.check_call("%s %s -fin %s -fout %s"%(pyexe,sosexe,iimg,oimg),shell=True)

# Read in smoothed file
saxes,smt = sep.read_file(oimg)
smt = smt.reshape(saxes.n,order='F').T
smt = np.transpose(smt,(0,2,1))
#viewcube3d(smt,width3=1.0)

print(smt.shape)
tptch = normextract(smt,nzp=64,nxp=64)
print(tptch.shape)

for itr in range(ntr):
  img = hf[keys[itr    ]][0,0]
  lbl = hf[keys[itr+ntr]][0,0]
  viewcube3d(tptch[np.random.randint(0,105)],show=False)
  viewcube3d(img)
  #plot_cubeiso(tptch[np.random.randint(0,105)],elev=15,verb=False,show=False,stack=False)
  #plot_cubeiso(img,elev=15,verb=False,show=True,stack=False)

