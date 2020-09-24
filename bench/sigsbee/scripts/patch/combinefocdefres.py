import h5py
import numpy as np
from deeplearn.dataloader import WriteToH5
from genutils.ptyprint import progressbar
from genutils.plot import plot_cubeiso

# Get the three input files
hff = h5py.File('/scr2/joseph29/sigsbee_fltfoc.h5','r')
hfr = h5py.File('/scr2/joseph29/sigsbee_fltreslbl_gpu.h5','r')
hfd = h5py.File('/scr2/joseph29/sigsbee_fltdeflbl_gpu.h5','r')

fkeys = list(hff.keys())
rkeys = list(hfr.keys())
dkeys = list(hfd.keys())

fnex = len(fkeys)//2
rnex = len(rkeys)//2
dnex = len(dkeys)//2

# Output file
wh5 = WriteToH5('/scr2/joseph29/sigsbee_focdefres.h5',dsize=1)

qc = False; j = 200
for iex in progressbar(range(rnex),"nex:"):
  # Get the examples
  xf = hff[fkeys[iex     ]][0,:,:,:,0]
  yf = hff[fkeys[iex+fnex]][0,0]
  xr = hfr[rkeys[iex     ]][0,0,:,:,:]
  yr = hfr[rkeys[iex+rnex]][0,0]
  xd = hfd[dkeys[iex     ]][0,0,:,:,:]
  yd = hfd[dkeys[iex+dnex]][0,0]
  if(qc and iex%j == 0):
    plot_cubeiso(xf,stack=True,elev=15,show=False,verb=False,title='Focused')
    plot_cubeiso(xr,stack=True,elev=15,show=False,verb=False,title='R-Defocused')
    plot_cubeiso(xd,stack=True,elev=15,show=True,verb=False,title='Defocused')
  # Write each to the output file
  wh5.write_examples(xf[np.newaxis],yf[np.newaxis])
  # Either write a defocused or residually-defocused
  if(np.random.choice([0,1])):
    wh5.write_examples(xr[np.newaxis],yr[np.newaxis])
  else:
    wh5.write_examples(xd[np.newaxis],yd[np.newaxis])

# Close the files
hff.close(); hfr.close(); hfd.close()

