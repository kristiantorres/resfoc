import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import normextract, resample, plot_seglabel
from deeplearn.dataloader import WriteToH5
from genutils.ptyprint import create_inttag, progressbar
from genutils.plot import plot_img2d
import subprocess

# IO
sep = seppy.sep()

taxes = sep.read_header("hale_foctrimgs.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

os = [oa,oz,ox]; ds = [da,dz,dx]

# Size of input data for reading
nw = 20; nex = nm//nw
# Size of a single patch
#ptchz = 64; ptchx = 64
ptchz = 128; ptchx = 128

# Define window
bxw = 140; exw = 652
bzw = 100; ezw = 356

# Open the output HDF5 file
wh5 = WriteToH5("/scr2/joseph29/hale2_fltseg128.h5",dsize=1)

k = 0
for iex in progressbar(range(nex),"iex:"):
  # Read in the focused images
  faxes,foc = sep.read_wind("hale_foctrimgs.H",fw=k,nw=nw)
  foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct  = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3)))
  focts = np.sum(foct,axis=1)
  # Read in the labels
  laxes,lbl = sep.read_wind("hale_trlbls.H",fw=k,nw=nw)
  lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
  ldats = []; llbls = []
  for iw in range(nw):
    # Window the image and label and resample
    focxw  = focts[iw,bxw:exw,:]
    lblw  = lbl   [iw,bxw:exw,bzw:ezw]
    focxg  = agc(focxw)
    focg = focxg[:,bzw:ezw]
    # Write the label and the image and smooth
    sep.write_file("imgw.H",focg.T)
    #sep.write_file("lblw.H",lblw.T)
    #sp = subprocess.check_call("python scripts/SOSmoothing.py -fin imgw.H -labels lblw.H -fout smtw.H",shell=True)
    sp = subprocess.check_call("python scripts/SOSmoothing.py -fin imgw.H -fout smtw.H",shell=True)
    saxes,smt = sep.read_file("smtw.H")
    smt = smt.reshape(saxes.n,order='F').T
    smtw = smt [:512,:]
    # Transpose
    smtwt = np.ascontiguousarray(smtw.T)
    lblwt = np.ascontiguousarray(lblw.T)
    #plot_img2d(smtwt)
    # Extract patches
    fptch = normextract(smtwt,nzp=ptchz,nxp=ptchx,norm=True)
    lptch = normextract(lblwt,nzp=ptchz,nxp=ptchx,norm=False)
    #plot_img2d(fptch[0],dx=dx,dz=dz,show=False)
    #plot_seglabel(fptch[0],lptch[0],show=True)
    nptch = lptch.shape[0]
    # Write the training data to HDF5 file
    wh5.write_examples(fptch,lptch)
  k += nw

