import inpout.seppy as seppy
import numpy as np
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
from genutils.movie import viewresangptch

sep = seppy.sep()

# Read in image
iaxes,img = sep.read_file("resmskoverwt.H")
[nz,na,nx,nro] = iaxes.n; [dz,da,dx,dro] = iaxes.d; [oz,oa,ox,oro] = iaxes.o
img  = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
imgt = np.ascontiguousarray(np.transpose(img, (0,2,3,1)))

# Read in semblance and predictions
paxes,prds = sep.read_file("torchprdsptch.H")
prds = prds.reshape(paxes.n,order='F').T

saxes,smbs = sep.read_file("sembptch.H")
smbs = smbs.reshape(saxes.n,order='F').T

nzp = 64; nxp = 64
strdz = int(nzp/2 + 0.5)
strdx = int(nxp/2 + 0.5)

# Define window
bxw = 20;  exw = 480
bzw = 240; ezw = 1150

# Window and create patches
imgw = imgt[:,:,bzw:ezw,bxw:exw]
pea = PatchExtractor((nro,na,nzp,nxp),stride=(nro,na,strdz,strdx))
aptch = np.squeeze(pea.extract(imgw))

print(aptch.shape)

for i in range(10,27):
  for j in range(2,5):
    print("Patch (%d,%d)"%(i,j))
    viewresangptch(aptch[i,j],prds[i,j],oro,dro,smb=smbs[i,j])

