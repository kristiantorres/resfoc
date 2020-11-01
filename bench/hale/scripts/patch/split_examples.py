import inpout.seppy as seppy
import os,glob
import numpy as np
from resfoc.gain import agc
from genutils.ptyprint import create_inttag
from genutils.plot import plot_img2d
from deeplearn.utils import plot_seglabel

sep = seppy.sep()

taxes = sep.read_header("hale_foctrimgscat.H")
[nz,na,naz,nx,nm] = taxes.n
[dz,da,daz,dx,dm] = taxes.d
[oz,oa,oaz,ox,om] = taxes.o

# Number of examples to be read in at once
nw = 20; nex = nm//nw

# Window to apply
begx,endx = 140,652
begz,endz = 100,356

# Get the last example written in the output directory
odir = './dat/split/'
dpath = '/net/thing/scr2/joseph29/hale_split/'

files = sorted(glob.glob(odir+"*.H"))
if(len(files) == 0):
  lnum = k = ctr = 0
else:
  lnum = int(os.path.splitext(files[-1])[0].split('lbl-')[1])
  ctr = lnum + 1
  k = (lnum//nw+1)*nw

for iex in range(nex):
  # Read in the images
  faxes,foc = sep.read_wind("hale_foctrimgscat.H",fw=k,nw=nw)
  foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct  = np.ascontiguousarray(np.transpose(foc[:,begx:endx,0,:,begz:endz],(0,2,1,3))) # [nw,nx,na,nz] -> [nw,na,nx,nz]
  # Read in the fault labels
  laxes,lbl = sep.read_wind("hale_trlblscat.H",fw=k,nw=nw)
  lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
  lbl = lbl[:,begx:endx,begz:endz]
  for iw in range(nw):
    foctstk = agc(np.sum(foct,axis=1))
    #plot_img2d(foctstk[iw].T,pclip=0.5,show=False)
    #plot_seglabel(foctstk[iw].T,lbl[iw].T,pclip=0.5,show=True)
    # Write out the image and the label to files
    tag = create_inttag(ctr,10000)
    sep.write_file(odir+"img-"+tag+".H",foctstk[iw].T,dpath=dpath)
    sep.write_file(odir+"lbl-"+tag+".H",lbl[iw].T,dpath=dpath)
    ctr += 1
  k += nw

