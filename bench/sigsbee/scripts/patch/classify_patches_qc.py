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

#def focdefcomp(dstk,fstk,lbl,dprd,frpd):


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

defout = []
qc = True
for iex in progressbar(range(nex),"nex:",verb=False):
  deffound = False
  idx = np.random.randint(nex)
  # Focused
  cubf = hffoc[fockeys[idx]][0,:,:,:,0]
  fstk = fstks[idx,:,:,0]
  # Defocused
  cubd = hfdef[fockeys[idx]][0,:,:,:,0]
  dstk = dstks[idx,:,:,0]
  # Label and prediction
  lbl  = lbls [idx,:,:,0]
  dprd = dprds[idx,:,:,0]
  fprd = fprds[idx,:,:,0]
  # Compute fault metrics
  fltnum = np.sum(lbl)
  # If example has faults, use fault criteria
  print("Example %d Faultpix=%d"%(idx,fltnum))
  # Image metrics
  corrimg = corrsim(fstk,dstk); corrprb = corrsim(fprd,dprd)
  print("Corrimg=%f Corrprb=%f"%(corrimg,corrprb))
  fivar = varimax(fstk); divar = varimax(dstk)
  ivarrat = divar/fivar
  print("Fivar=%f Divar=%f Rat=%f"%(fivar,divar,ivarrat))
  fpvar = varimax(fprd); dpvar = varimax(dprd)
  pvarrat = dpvar/fpvar
  print("Fpvar=%f Dpvar=%f Rat=%f"%(fpvar,dpvar,pvarrat))
  # Angle metrics
  fsemb = semblance_power(cubf[31:])
  dsemb = semblance_power(cubd[31:])
  sembrat = dsemb/fsemb
  print("Fsemb=%f Dsemb=%f Rat=%f"%(fsemb,dsemb,sembrat))
  if(sembrat < 0.4):
    print("Defocused Fault/Angle")
    defout.append(cubd)
    deffound = True
  if(fltnum > pixthresh and not deffound):
    if(sembrat < thresh1 and pvarrat  < thresh1):
      print("Defocused fault")
      defout.append(cubd)
    elif(sembrat < thresh2 or pvarrat < thresh2):
      print("Defocused fault")
      defout.append(cubd)
    #print("Corrimg=%f Corrprb=%f"%(corrimg,corrprb))
    #if(corrimg < thresh1 and pvarrat < thresh1):
    #  print("Defocused fault")
    #  defout.append(cubd)
    #elif(corrimg < thresh2 or pvarrat < thresh2):
    #  print("Defocused fault")
    #  defout.append(cubd)
  elif(not deffound):
    # Compute angle metrics
    #print("Fsemb=%f Dsemb=%f Ratio=%f"%(fsemb,dsemb,sembrat))
    if(sembrat < thresh3):
      print("Defocused angle")
      #plot_cubeiso(cubd,stack=True,elev=15,show=False,verb=False)
      #plot_cubeiso(cubf,stack=True,elev=15,show=True,verb=False)
      defout.append(cubd)
  if(qc and iex%1 == 0):
    plot_cubeiso(cubf[31:],stack=True,elev=15,show=False,verb=False,title='Focused')
    plot_cubeiso(cubd[31:],stack=True,elev=15,show=False,verb=False,title='Defocused')
    fig,axarr = plt.subplots(1,5,figsize=(15,3))
    axarr[0].imshow(fstk,cmap='gray',interpolation='bilinear',aspect='auto')
    axarr[0].set_title('Focused')
    axarr[1].imshow(dstk,cmap='gray',interpolation='bilinear',aspect='auto')
    axarr[1].set_title('Defocused')
    axarr[2].imshow(lbl,cmap='jet',interpolation='none',aspect='auto')
    axarr[3].imshow(fprd,cmap='jet',interpolation='bilinear',aspect='auto')
    axarr[3].set_title('Focused')
    axarr[4].imshow(dprd,cmap='jet',interpolation='bilinear',aspect='auto')
    axarr[4].set_title('Defocused')
    plt.show()
  print("")

hfdef.close()
hffoc.close()

