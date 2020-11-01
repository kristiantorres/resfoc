import inpout.seppy as seppy
import os,glob
import numpy as np
from resfoc.gain import agc
from genutils.ptyprint import create_inttag, progressbar
from genutils.plot import plot_img2d
from genutils.movie import viewcube3d
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
odir = './dat/split_angs/'
dpath = '/net/thing/scr2/joseph29/hale_split_angs/'

files = sorted(glob.glob(odir+"*.H"))
lnum = k = ctr = 0
#if(len(files) == 0):
#  lnum = k = ctr = 0
#else:
#  lnum = int(os.path.splitext(files[-1])[0].split('lbl-')[1])
#  ctr = lnum + 1
#  k = (lnum//nw+1)*nw

aagc = True
for iex in progressbar(range(nex),"nex:"):
  # Read in the images
  faxes,foc = sep.read_wind("hale_foctrimgscat.H",fw=k,nw=nw)
  foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
  foct  = np.ascontiguousarray(np.transpose(foc[:,begx:endx,0,:,:],(0,2,1,3))) # [nw,nx,na,nz] -> [nw,na,nx,nz]
  # Defocused images
  daxes,dfc = sep.read_wind("hale_deftrimgscat.H",fw=k,nw=nw)
  dfc   = np.ascontiguousarray(dfc.reshape(daxes.n,order='F').T).astype('float32')
  dfct  = np.ascontiguousarray(np.transpose(dfc[:,begx:endx,0,:,:],(0,2,1,3))) # [nw,nx,na,nz] -> [nw,na,nx,nz]
  # Residually defocused image
  raxes,res = sep.read_wind("hale_restrimgscat.H",fw=k,nw=nw)
  res   = np.ascontiguousarray(res.reshape(raxes.n,order='F').T).astype('float32')
  rest  = np.ascontiguousarray(np.transpose(res[:,begx:endx,0,:,:],(0,2,1,3))) # [nw,nx,na,nz] -> [nw,na,nx,nz]
  laxes,lbl = sep.read_wind("hale_trlblscat.H",fw=k,nw=nw)
  lbl = np.ascontiguousarray(lbl.reshape(laxes.n,order='F').T).astype('float32')
  lbl = lbl[:,begx:endx,begz:endz]
  for iw in range(nw):
    foctg = agc(foct)[iw,:,:,begz:endz]
    dfctg = agc(dfct)[iw,:,:,begz:endz]
    restg = agc(rest)[iw,:,:,begz:endz]
    foctn = foct[iw,:,:,begz:endz]
    dfctn = dfct[iw,:,:,begz:endz]
    restn = rest[iw,:,:,begz:endz]
    # Read in the fault labels
    foctstk = agc(np.sum(foct,axis=1))[iw,:,begz:endz]
    deftstk = agc(np.sum(dfct,axis=1))[iw,:,begz:endz]
    reststk = agc(np.sum(rest,axis=1))[iw,:,begz:endz]
    #plot_img2d(foctstk.T,pclip=0.5,show=False)
    #plot_img2d(deftstk.T,pclip=0.5,show=False)
    #plot_img2d(reststk.T,pclip=0.5,show=False)
    #plot_seglabel(foctstk.T,lbl[iw].T,pclip=0.5,show=False)
    #viewcube3d(foctg[32:,:,:].T,width3=1.0)
    # Write out the image and the label to files
    tag = create_inttag(ctr,10000)
    sep.write_file(odir+"fstk-"+tag+".H",foctstk.T,dpath=dpath)
    sep.write_file(odir+"fimgg-"+tag+".H",foctg.T,dpath=dpath)
    sep.write_file(odir+"fimgn-"+tag+".H",foctn.T,dpath=dpath)
    sep.write_file(odir+"dstk-"+tag+".H",deftstk.T,dpath=dpath)
    sep.write_file(odir+"dimgg-"+tag+".H",dfctg.T,dpath=dpath)
    sep.write_file(odir+"dimgn-"+tag+".H",dfctn.T,dpath=dpath)
    sep.write_file(odir+"rstk-"+tag+".H",reststk.T,dpath=dpath)
    sep.write_file(odir+"rimgg-"+tag+".H",restg.T,dpath=dpath)
    sep.write_file(odir+"rimgn-"+tag+".H",restn.T,dpath=dpath)
    sep.write_file(odir+"lbl-"+tag+".H",lbl[iw].T,dpath=dpath)
    ctr += 1
  k += nw

