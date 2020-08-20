from deeplearn.dataloader import load_allssimdata
import numpy as np
import h5py
from genutils.ptyprint import create_inttag, progressbar
from genutils.movie import viewimgframeskey

allx,ally = load_allssimdata("/scr1/joseph29/ssimresdat64.h5",None,465)

[nex,nxp,nzp,nro] = allx.shape

ro1 = np.zeros(nro)
idx1 = int((nro-1)/2)
ro1[idx1] = 1.0

# Open a h5 dataset
hf = h5py.File("/scr1/joseph29/ssimresdat64clean.h5",'w')

# Loop over each example and remove 10000 ro=1
k = 0
for iex in progressbar(range(nex),"iex"):
  #print(ally[iex])
  #viewimgframeskey(allx[iex].T,transp=False)
  if((ally[iex] == ro1).all()):
    if(np.random.choice([0,0,0,1])):
      datatag = create_inttag(iex,nex)
      hf.create_dataset("x"+datatag, (nxp,nzp,nro), data=allx[iex], dtype=np.float32)
      hf.create_dataset("y"+datatag, (nro,), data=ally[iex], dtype=np.float32)
    else:
      k += 1
  else:
    datatag = create_inttag(iex,nex)
    hf.create_dataset("x"+datatag, (nxp,nzp,nro), data=allx[iex], dtype=np.float32)
    hf.create_dataset("y"+datatag, (nro,), data=ally[iex], dtype=np.float32)

hf.close()
