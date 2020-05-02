import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey
from resfoc.estro import onehot2rho

#hf = h5py.File('/scr2/joseph29/ssimresdat.h5')
#hf = h5py.File('/scr2/joseph29/ssimresdatwind.h5')
#hf = h5py.File('/scr1/joseph29/ssimresdat64.h5')
hf = h5py.File('/scr1/joseph29/ssimresdat64clean.h5')

nro = 19; oro = 0.9775; dro = 0.0025
rhos = np.linspace(oro,oro+(nro-1)*dro,nro)

keys = list(hf.keys())
nex = int(len(keys)/2)

print(hf[keys[0]].shape)

myex = 50; myp = 139
#print(hf[keys[myex+nex]][myp])
#print(np.arange(19))
#viewimgframeskey(hf[keys[myex]][myp].T,transp=False,ottl=oro,dttl=dro,ttlstring=r'$\rho=$%.3f',interp='sinc',show=False)
#viewimgframeskey(hf[keys[myex]][:,:,:,9],transp=True,interp='sinc')

ro1 = np.zeros(19)
ro1[9] = 1.0

print(ro1)

tot = np.zeros(nro)
for iex in range(nex):
  ilbl = hf[keys[iex+nex]]
  #tot += np.sum(ilbl,axis=0)
  tot += ilbl

hf.close()

labels = []
for iro in range(nro):
  labels.append("%.3f"%(rhos[iro]))

plt.figure(1,figsize=(12,6))
plt.bar(np.arange(nro),tot,bottom=rhos)
plt.xticks(np.arange(nro), labels)
plt.xlabel(r'$\rho$')
plt.show()

