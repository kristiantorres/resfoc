import inpout.seppy as seppy
import numpy as np
from resfoc.gain import agc
from deeplearn.utils import normextract, resample, plotseglabel
from deeplearn.focuslabels import corrsim, semblance_power
import h5py
import matplotlib.pyplot as plt
from genutils.plot import plot_cubeiso

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
ptchz = 64; ptchx = 64

# Define window
bxw = 20;  exw = nx - 20
bzw = 177; ezq = nz

defout = []
pixthresh = 20
thresh1 = 0.7; thresh3 = 0.7

k = 0
for iex in range(nex):
  # Read in the focused images
  faxes,foc = sep.read_wind("sigsbee_foctrimgs.H",fw=k,nw=nw)
  foc  = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct = np.ascontiguousarray(np.transpose(foc[:,:,0,:,:],(0,2,1,3)))
  # Read in defocused images
  daxes,dfc = sep.read_wind("sigsbee_deftrimgs.H",fw=k,nw=nw)
  dfc  = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
  dfct = np.ascontiguousarray(np.transpose(dfc[:,:,0,:,:],(0,2,1,3)))
  # Read in the labels
  laxes,lbl = sep.read_wind("sigsbee_trlblsint.H",fw=k,nw=nw)
  lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
  #TODO: This should be a patching worker, distributed over a cluster
  for iw in range(nw):
    # Window the image and label and resample
    focw  = foct[iw,:,bxw:exw,bzw:nz]
    dfcw  = dfct[iw,:,bxw:exw,bzw:nz]
    lblw  = lbl [iw,bxw:exw,bzw:nz]
    # Extract patches
    fptch = normextract(focw,nzp=ptchz,nxp=ptchx,norm=True)
    dptch = normextract(dfcw,nzp=ptchz,nxp=ptchx,norm=True)
    lptch = normextract(lblw,nzp=ptchz,nxp=ptchx,norm=False)
    # Transpose the patches
    fptcht = np.ascontiguousarray(np.transpose(fptch,(0,1,3,2)))
    dptcht = np.ascontiguousarray(np.transpose(dptch,(0,1,3,2)))
    lptcht = np.ascontiguousarray(np.transpose(lptch,(0,2,1)))
    # Create image stacks
    fsptch = np.sum(fptcht,axis=1); dsptch = np.sum(dptcht,axis=1)
    nptch = fptch.shape[0]
    for ip in range(nptch):
      # Compute fault metrics
      fltnum = np.sum(lptcht[ip])
      # If example has faults, use fault criteria
      if(fltnum > pixthresh):
        corrimg = corrsim(dsptch[ip],fsptch[ip])
        if(corrimg < thresh1):
          print("Defocused fault - %.3f"%(corrimg))
          plot_cubeiso(dptcht[ip],os=os,ds=ds,stack=True,elev=15,show=False,verb=False)
          plot_cubeiso(fptcht[ip],os=os,ds=ds,stack=True,elev=15,show=True,verb=False)
          defout.append(dptcht[ip])
    else:
      # Compute angle metrics
      fsemb = semblance_power(fptch[ip])
      dsemb = semblance_power(dptch[ip])
      sembrat = dsemb/fsemb
      if(sembrat < thresh3):
        print("Defocused ang")
        plot_cubeiso(dptcht[ip],os=os,ds=ds,stack=True,elev=15,show=False,verb=False)
        plot_cubeiso(fptcht[ip],os=os,ds=ds,stack=True,elev=15,show=True,verb=False)
        defout.append(dptcht[ip])
    # Write the training data to HDF5 file
  k += nw

