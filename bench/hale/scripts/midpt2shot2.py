import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in Bobs for comparison
# For some reason there is a 2m discrepancy between source coordinates
# This is consistent across all coordinates
baxes,bob = sep.read_file("stack.H")
bob = bob.reshape(baxes.n,order='F')
[nt,nrxb,nsxb] = baxes.n; [ot,orxb,osxb] = baxes.o; [dt,drxb,dsxb] = baxes.d

bsxs = np.linspace(osxb,osxb + (nsxb-1)*dsxb,nsxb)

maxes,dat = sep.read_file("./dat/allmidptswin.HH")
[nt,no,nm] = maxes.n; [ot,oo,om] = maxes.o; [dt,do,dm] = maxes.d
dat = dat.reshape(maxes.n,order='F').T

# Select every other midpoint
a = dat[::2,:,:]
noa = a.shape[1]; ooa = oo; doa = do
nma = a.shape[0]; oma = om; dma = dm*2

# Flatten along midpoint and compute coordinates
mida = np.linspace(oma,oma+(nma-1)*dma,nma)
offa = np.linspace(ooa,ooa+(noa-1)*doa,noa)
midsalla = np.zeros([nma,noa],dtype='float32')
offsalla = np.zeros([nma,noa],dtype='float32')

for im in range(nma):
  midsalla[im,:] = mida[im]
  offsalla[im,:] = offa[:]

af     = a.reshape([nma*noa,nt])
midaf = midsalla.reshape([nma*noa])
offaf = offsalla.reshape([nma*noa])

b = dat[1::2,:,:]
nob = b.shape[1]; oob = oo + do/2; dob = do
nmb = b.shape[0]; omb = om + dm*2; dmb = dm*2
oob = oo + do/2

midb = np.linspace(omb,omb+(nmb-1)*dmb,nmb)
offb = np.linspace(oob,oob+(nob-1)*dob,nob)
midsallb = np.zeros([nmb,nob],dtype='float32')
offsallb = np.zeros([nmb,nob],dtype='float32')

for im in range(nmb):
  midsallb[im,:] = midb[im]
  offsallb[im,:] = offb[:]

bf = b.reshape([nmb*nob,nt])
midbf = midsallb.reshape([nmb*nob])
offbf = offsallb.reshape([nmb*nob])

threeD = np.concatenate([af,bf],axis=0)
mids3D = np.concatenate([midaf,midbf],axis=0)
offs3D = np.concatenate([offaf,offbf],axis=0)

ntr = threeD.shape[0]

#for itr in range(ntr):
#  print("itr=%d offset=%f cmp_x=%f"%(itr+1,offs3D[itr],mids3D[itr]))

shtsall = (mids3D - offs3D/2.0) - 0.002
recsall = (mids3D + offs3D/2.0) - 0.002

#plt.figure()
#plt.plot(shtsall)
#plt.show()

# Form a shot grid
nsx = 173; osx = np.min(shtsall); dsx = np.abs(shtsall[1] - shtsall[0])
print("Shot grid: nsx=%d osx=%f dsx=%f"%(nsx,osx,dsx))

bsx = np.zeros(len(shtsall),dtype='float32')
cnt = np.zeros(nsx,dtype='int32')

for isx in range(len(shtsall)):
  smp = int((shtsall[isx] - osx)/dsx + 0.5)
  if(smp > nsx-1): smp = nsx - 1
  bsx[isx] = smp*dsx + osx
  cnt[smp] += 1

srcx = np.unique(bsx)
srcxw = srcx[0:]
nsxw = len(srcxw)

fig = plt.figure(); ax = fig.gca()
ax.plot(cnt,linewidth=2)
ax.set_xlabel('Shot number',fontsize=15)
ax.set_ylabel('Receivers per shot',fontsize=15)
ax.tick_params(labelsize=15)
plt.show()

# Read in velocity model
vaxes,vel = sep.read_file("vintz.H")
[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o
vel = vel.reshape(vaxes.n,order='F')

zeros = np.zeros(nsxw) + 10*dz
fig = plt.figure(figsize=(14,7)); ax = fig.gca()
ax.imshow(vel,cmap='jet',extent=[ox,ox+nx*dx,nz*dz,0])
ax.set_xlabel('X (km)',fontsize=15)
ax.set_ylabel('Z (km)',fontsize=15)
ax.tick_params(labelsize=15)
ax.scatter(srcxw,zeros,c='tab:red',marker='*')
ax.scatter(bsxs,zeros,c='tab:blue',marker='*')
plt.show()

# Form a receiver grid (before isx = 23 does not make sense)
nrx = 48; orx = 0.264; drx = dsx
sort = True
nrecs =  np.zeros(nsxw,dtype='float32')
shots = []; recx = []; srcxo = []; nrecso = []; afsht = 24
# Get receivers for each shot
for isx in range(nsxw):
  sidxs = bsx == srcxw[isx]
  # Get receiver coords
  recs = np.asarray(recsall[sidxs])
  nrecs[isx] = len(recs)
  isht = np.asarray(threeD[sidxs,:])
  if(sort):
    idxs = np.argsort(recs)
    recs = recs[idxs]
    isht = isht[idxs,:]
  offs = recs - srcxw[isx]
  obin = np.zeros(nrx,dtype='float32')
  ocnt = np.zeros(nrx,dtype='int32')
  for ioff in range(len(offs)):
    rmp = int((offs[ioff] - orx)/drx + 0.5)
    if(rmp > nrx-1): rmp = nrx - 1
    obin[ioff] = rmp*drx + orx
    ocnt[rmp] += 1
  if(isx > afsht):
    recso = srcxw[isx] + obin[:int(nrecs[isx])]
    nrecso.append(nrecs[isx])
    srcxo.append(srcxw[isx])
    recx.append(recso)
    mut = np.squeeze(mute(isht,dt=dt,dx=dsx,v0=1.4,t0=0.2,half=False))
    shots.append(mut)

shtsout = np.concatenate(shots,axis=0)
recxs = np.concatenate(recx,axis=0)
srcxo = np.asarray(srcxo)
nrecso = np.asarray(nrecso)

sep.write_file("hale_shotflatsort2.H",shtsout.T,ds=[dt,1.0])
sep.write_file("hale_srcxflat2.H",srcxo)
sep.write_file("hale_recxflatsort2.H",recxs)
sep.write_file("hale_nrec2.H",nrecso)

