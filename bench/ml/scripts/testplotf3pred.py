import numpy as np
import h5py
from deeplearn.dataloader import load_allflddata
from deeplearn.utils import thresh, plotseglabel
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor

with h5py.File('./dat/ep0.h5','r') as hf:
  pred1 = hf['pred'][:]

f3dat = load_allflddata('./dat/f3tstdat.h5',105)

dummy = np.zeros([512,1024])
psize = (128,128)
ssize=(64,64)
pe = PatchExtractor(psize,stride=ssize)
dptch = pe.extract(dummy)

iimg = f3dat[105:210,:,:]
iimg = iimg.reshape([7,15,128,128])
rimg = pe.reconstruct(iimg)

iprd = pred1[105:210,:,:]
iprd = iprd.reshape([7,15,128,128])
rprd = pe.reconstruct(iprd)
tprd = thresh(rprd,0.9)

plotseglabel(rimg[200:,:],tprd[200:,:],color='blue',
             xlabel='Inline',ylabel='Time (s)',xmin=0.0,xmax=(1023)*25/1000.0,
             zmin=(200-1)*0.004,zmax=(511)*0.004,vmin=-2.5,vmax=2.5,aspect=6.5,show=True)


