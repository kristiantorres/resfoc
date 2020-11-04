import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from joblib import Parallel, delayed
import subprocess
import torch
from deeplearn.torchnets import Unet, Vgg3a_3d
from deeplearn.torchpredict import segmentfaults
from deeplearn.utils import plot_segprobs, plot_seglabel, thresh
from resfoc.estro import estro_fltangfocdefoc, refocusimg
from genutils.plot import plot_img2d, plot_vel2d, plot_rhoimg2d

sep = seppy.sep()
# Read in the prestack residually migrated image
#iaxes,img = sep.read_file("resmigtrntsos.H")
iaxes = sep.read_header("resmigtrntsos.H")
[nz,na,nx,nro] = iaxes.n; [dz,da,dx,dro] = iaxes.d; [oz,oa,ox,oro] = iaxes.o
#img  = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')
# Apply AGC
#gimg  = np.asarray(Parallel(n_jobs=24)(delayed(agc)(img[iro]) for iro in range(nro)))
#gimgt = np.ascontiguousarray(np.transpose(gimg,(0,2,3,1)))
#gimgtw = gimgt[:,32:,100:356,:]
#sep.write_file("estro_preproc.H",gimgtw.T)
naxes,gimgtw = sep.read_file("estro_preproc.H")
gimgtw = np.ascontiguousarray(gimgtw.reshape(naxes.n,order='F').T)

# Read in the stack
saxes,stk = sep.read_file("resmigtrnstk.H")
stk = np.ascontiguousarray(stk.reshape(saxes.n,order='F').T).astype('float32')

# Apply AGC to stack
gstk = agc(stk)
gstkw = gstk[:,100:356]

# Smooth stack
#sep.write_file("presmooth.H",gstkw.T)
#sp = subprocess.check_call("python scripts/SOSmoothing.py -fin presmooth.H -fout smooth.H",shell=True)
naxes,smt = sep.read_file("smooth.H")
smt = smt.reshape(naxes.n,order='F')

# Read in the network
unet = Unet()
device = torch.device('cpu')
unet.load_state_dict(torch.load('/scr1/joseph29/hale2_fltsegsm.pth',map_location=device))

# Segment faults
ptchz,ptchx = 128,128
iprb = segmentfaults(smt,unet,nzp=ptchz,nxp=ptchx)
plot_segprobs(smt,iprb,pmin=0.5,show=False)
ilbl = thresh(iprb,0.5)
plot_seglabel(smt,ilbl)

# Read in the fault focusing network
fnet = Vgg3a_3d()
device = torch.device('cuda:1')
fnet.load_state_dict(torch.load('/scr1/joseph29/hale2_fltfoc.pth',map_location=device))
fnet.to(device)

# Estimate rho
rho,fltfocs,fltprds = estro_fltangfocdefoc(gimgtw,fnet,dro,oro,fltlbls=ilbl,device=device,verb=True)

plot_rhoimg2d(smt,rho,dx=dx,dz=dz,ox=ox,oz=100*dz,aspect=2.0)

# Plot rho img and refocus
sep.write_file("beitorch_fltprds.H",fltprds.T)
sep.write_file("beitorch_rho.H",rho,ds=[dz,dx],os=[oz,ox])

