import numpy as np
from deeplearn.dataloader import load_alldata
import inpout.seppy as seppy

sep = seppy.sep([])

xall,yall = load_alldata("/scr2/joseph29/resmigsmall_tr.h5","/scr2/joseph29/resmigsmall_va.h5",10)

print(xall.shape,yall.shape)
ntot = xall.shape[0]

# Make a random array of ints for plotting
probes = np.unique(np.random.randint(0,ntot,10))

movargs = 'gainpanel=a pclip=100'
rhoargs = 'color=j newclip=1 wantscalebar=y'

for ipr in probes:
  # Find the file and the example for the probe
  print("Example %d"%(ipr))
  # Choose a random value within the batch
  sep.pltgreymovie(xall[ipr,:,:,:],greyargs=movargs,o3=0.95,d1=20,d2=20,d3=0.01,bg=True)
  sep.pltgreyimg(yall[ipr,:,:,0],greyargs=rhoargs,d1=20,d2=20,bg=False)

