import h5py
import numpy as np
import matplotlib.pyplot as plt

hf = h5py.File("/scr1/joseph29/faultpatch.h5",'r')

keys = list(hf.keys())

nex = int(len(keys)/2)
nptch = hf[keys[0]].shape[0]

for iex in range(nex):
  for iptch in range(nptch):
    plt.imshow(hf[keys[iex]][iptch,:,:,0],cmap='gray')
    print(hf[keys[iex+nex]][iptch][0])
    plt.show()

