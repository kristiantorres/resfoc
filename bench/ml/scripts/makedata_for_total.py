import inpout.seppy as seppy
import h5py
import numpy as np
from deeplearn.utils import resizepow2
from deeplearn.dataloader import load_all_unlabeled_data, load_labeled_flat_data
from sklearn.utils import shuffle
import random
from utils.ptyprint import create_inttag, progressbar

sep = seppy.sep()

# F3 dataset
iaxes,img = sep.read_file("./dat/f3cube.H")
img = img.reshape(iaxes.n,order='F')
img = np.ascontiguousarray(img.T).astype('float32')

iline = resizepow2(img[11]).T

np.save("f3iline.npy",iline)

# Angle gather focus training data
# Load all data
nex = 5040
nimgs = int(nex/105)
focdat = load_all_unlabeled_data("/scr1/joseph29/angfaultfocptch.h5",0,nimgs)
resdat = load_all_unlabeled_data("/scr1/joseph29/angfaultresptch.h5",0,nimgs)
defdat,deflbl = load_labeled_flat_data("/scr1/joseph29/alldefocangs.h5",None,0,nex)

## Size of each dataset
nfoc = focdat.shape[0]
nres = resdat.shape[0]
ndef = defdat.shape[0]
ndiff = nfoc - ndef

# Make the three datasets the same size
idxs1 = random.sample(range(nfoc), ndiff)
idxs2 = random.sample(range(nres), ndiff)

# Delete images randomly
foctrm = np.delete(focdat,idxs1,axis=0)
restrm = np.delete(resdat,idxs2,axis=0)

## Remove half of each defocused and combine
didxs1 = random.sample(range(ndef), int(ndef/2))
didxs2 = random.sample(range(ndef), int(ndef/2))

reshlf =  np.delete(restrm,didxs1,axis=0)
defhlf =  np.delete(defdat,didxs2,axis=0)

deftot = np.concatenate([reshlf,defhlf],axis=0)

# Create labels for defocused and focused
deflbls = np.zeros(ndef)
foclbls = np.ones(ndef)

# Concatenate focused and defocused images and labels
allx = np.concatenate([deftot,foctrm],axis=0)
ally = np.concatenate([deflbls,foclbls],axis=0)

allx,ally = shuffle(allx,ally,random_state=1992)

ntot = allx.shape[0]; na = allx.shape[1]; nz = allx.shape[2]; nx = allx.shape[2]

# Write the labeled data to file
hf = h5py.File('/scr1/joseph29/angfocus.h5','w')

for iex in progressbar(range(ntot), "nex:"):
  datatag = create_inttag(iex,ntot)
  hf.create_dataset("x"+datatag, (na,nx,nz,1), data=allx[iex], dtype=np.float32)
  hf.create_dataset("y"+datatag, (1,), data=ally[iex], dtype=np.float32)

hf.close()

# Angle mask
maxes,msk = sep.read_file("../focdat/mask.H")
msk = msk.reshape(maxes.n,order='F')
np.save('./dat/mask.npy',msk)

