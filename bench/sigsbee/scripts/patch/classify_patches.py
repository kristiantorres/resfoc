import os
import h5py
import numpy as np
import tensorflow as tf
from deeplearn.dataloader import load_alldata, WriteToH5
from tensorflow.keras.models import model_from_json
from genutils.ptyprint import progressbar
from genutils.plot import plot_cubeiso
from deeplearn.focuslabels import corrsim, semblance_power, varimax
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# Read in the stacked defocused and focused images
dstks,lbls = load_alldata("/net/thing/scr2/joseph29/sigsbee_defseg.h5",None,1)
fstks,lbls = load_alldata("/net/thing/scr2/joseph29/sigsbee_focseg.h5",None,1)

# Read in the network
with open('./dat/fltsegarch.json','r') as f:
  mdl = model_from_json(f.read())
mdl.load_weights('/scr1/joseph29/sigsbee_fltseg-chkpt.h5')

# Segment the faults on these images
dprds = mdl.predict(dstks,verbose=True)
fprds = mdl.predict(fstks,verbose=True)

# Loop over each cube and classify using the criteria
hfdef = h5py.File('/net/thing/scr2/joseph29/sigsbee_fltdef.h5','r')
hffoc = h5py.File('/net/thing/scr2/joseph29/sigsbee_fltfoc.h5','r')
fockeys = list(hffoc.keys())
nex = len(fockeys)//2

if(nex != dstks.shape[0]):
  raise Exception('should be same number of cubes and stacks')

pixthresh = 50
thresh1 = 0.7
thresh2 = 0.5
thresh3 = 0.7

wh5 = WriteToH5("/net/thing/scr2/joseph29/sigsbee_fltdeflbl.h5",dsize=1)
flbl = np.zeros([1],dtype='float32')

defout = []
qc = True
for iex in progressbar(range(nex),"nex:",verb=True):
  # Focused
  cubf = hffoc[fockeys[iex]][0,:,:,:,0]
  fstk = fstks[iex,:,:,0]
  # Defocused
  cubd = hfdef[fockeys[iex]][0,:,:,:,0]
  dstk = dstks[iex,:,:,0]
  # Label and prediction
  lbl  = lbls [iex,:,:,0]
  dprd = dprds[iex,:,:,0]
  fprd = fprds[iex,:,:,0]
  # Compute fault metrics
  fltnum = np.sum(lbl)
  fpvar = varimax(fprd); dpvar = varimax(dprd)
  pvarrat = dpvar/fpvar
  # Angle metrics
  fsemb = semblance_power(cubf[31:])
  dsemb = semblance_power(cubd[31:])
  sembrat = dsemb/fsemb
  if(sembrat < 0.4):
    #defout.append(cubd)
    wh5.write_examples(cubd[np.newaxis],flbl)
  # If example has faults, use fault criteria
  if(fltnum > pixthresh):
    if(sembrat < thresh1 and pvarrat  < thresh1):
      #defout.append(cubd)
      wh5.write_examples(cubd[np.newaxis],flbl)
    elif(sembrat < thresh2 or pvarrat < thresh2):
      #defout.append(cubd)
      wh5.write_examples(cubd[np.newaxis],flbl)
  else:
    # Compute angle metrics
    if(sembrat < thresh3):
      wh5.write_examples(cubd[np.newaxis],flbl)
      #defout.append(cubd)

#defarr = np.asarray(defout)
#print(defarr.shape)


hfdef.close()
hffoc.close()

