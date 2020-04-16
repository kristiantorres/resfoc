import numpy as np
import h5py
import matplotlib.pyplot as plt

#hf = h5py.File('/scr2/joseph29/ssimresdat.h5')
hf = h5py.File('/scr2/joseph29/ssimresdatwind.h5')

keys = list(hf.keys())
nex = int(len(keys)/2)
nro = 19; oro = 0.9775; dro = 0.0025
rhos = np.linspace(oro,oro+(nro-1)*dro,nro)

tot = np.zeros(nro)

for iex in range(nex):
  ilbl = hf[keys[iex+nex]]
  tot += np.sum(ilbl,axis=0)

hf.close()

print(rhos)
print(tot)
