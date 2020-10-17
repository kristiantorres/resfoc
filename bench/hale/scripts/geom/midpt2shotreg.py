import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

maxes,dat = sep.read_file("midmute.H")
[nt,nh,nm] = maxes.n; [ot,oh,om] = maxes.o; [dt,dh,dm] = maxes.d

dat = np.ascontiguousarray(dat.reshape(maxes.n,order='F').T)
datr = dat.reshape([nm*nh,nt])

mids = np.linspace(om,om+(nm-1)*dm,nm)
offs = np.linspace(oh,oh+(nh-1)*dh,nh)

midsall = np.zeros([nm,nh],dtype='float32')
offsall = np.zeros([nm,nh],dtype='float32')

for im in range(nm):
  midsall[im,:] = mids[im]
  offsall[im,:] = offs[:]

midsflt = midsall.flatten()
offsflt = offsall.flatten()

shtsall = (midsflt - offsflt)
recsall = (midsflt + offsflt)

# Form a shot grid
nsx = 174; osx = np.min(shtsall); dsx = 0.0335
dsx *= 2
bsx = np.zeros(len(shtsall),dtype='float32')
cnt = np.zeros(nsx,dtype='int32')

for isx in range(len(shtsall)):
  smp = int((shtsall[isx] - osx)/dsx + 0.6)
  if(smp > nsx-1): smp = nsx - 1
  bsx[isx] = smp*dsx + osx
  cnt[smp] += 1

srcx = np.unique(bsx)
srcxw = srcx[26:]
nsxw = len(srcxw)

# Read in velocity model
vaxes,vel = sep.read_file("vintz.H")
[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o
vel = vel.reshape(vaxes.n,order='F')

nrecs =  np.zeros(nsxw,dtype='int32')
for isx in range(nsxw):
  sidxs = bsx == srcxw[isx]
  recs = np.asarray(recsall[sidxs])
  nrecs[isx] = len(recs)

nrecmax = np.max(nrecs)
shots = np.zeros([nsxw,nrecmax,nt],dtype='float32')
recx = np.zeros([nsxw,nrecmax],dtype='float32')
# Get receivers for each shot
for isx in range(nsxw):
  sidxs = bsx == srcxw[isx]
  # Get receiver coords
  recs = np.asarray(recsall[sidxs])
  isht = np.asarray(datr[sidxs,:])
  # Sort the receivers
  idxs = np.argsort(recs)
  # Assign to the output array
  shots[isx,:nrecs[isx],:] = isht[idxs]
  recx[isx,:nrecs[isx]] = recs[idxs]

drx = 0.0335*2
sep.write_file("hale_shot.H",shots.T,ds=[dt,drx,dsx],os=[0,0,srcxw[0]])
sep.write_file("hale_srcx.H",srcxw)
sep.write_file("hale_recx.H",recx.T)

