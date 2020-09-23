import h5py
import numpy as np
from deeplearn.utils import plotseglabel
import matplotlib.pyplot as plt

hf1 = h5py.File('/scr2/joseph29/sigsbee_fltseg.h5','r')
keys1 = list(hf1.keys())
nex1 = len(keys1)//2

hf2 = h5py.File('/scr2/joseph29/sigsbee_focseg.h5','r')
keys2 = list(hf2.keys())
nex2 = len(keys2)//2

print(nex1,nex2)

j = 1000
for iex in range(nex1):
  if(iex%j == 0):
    idx = np.random.randint(nex1)
    print("idx=%d"%(idx))
    img1 = hf1[keys1[idx     ]][0,:,:,0]
    lbl1 = hf1[keys1[idx+nex1]][0,:,:,0]
    img2 = hf2[keys2[idx     ]][0,:,:,0]
    lbl2 = hf2[keys2[idx+nex2]][0,:,:,0]
    plt.imshow(img1-img2,cmap='gray',interpolation='bilinear'); plt.show()
    plotseglabel(img1,lbl1,aspect='auto',show=False)
    plotseglabel(img2,lbl2,aspect='auto',show=True)

hf1.close()
hf2.close()

