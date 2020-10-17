import inpout.seppy as seppy
import numpy as np
from oway.mute import mute
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

#print(np.min(shtsall),np.max(shtsall))
#print(np.min(recsall),np.max(recsall))

# Form a shot grid
nsx = 87; osx = np.min(shtsall); dsx = 0.0335
dsx *= 4
bsx = np.zeros(len(shtsall),dtype='float32')
cnt = np.zeros(nsx,dtype='int32')

for isx in range(len(shtsall)):
  smp = int((shtsall[isx] - osx)/dsx + 0.6)
  if(smp > nsx-1): smp = nsx - 1
  bsx[isx] = smp*dsx + osx
  cnt[smp] += 1

srcx = np.unique(bsx)
srcxw = srcx[13:]
nsxw = len(srcxw)

print("Shot grid: nsx=%d osx=%f dsx=%f"%(nsxw,srcx[13],dsx))

fig = plt.figure(); ax = fig.gca()
ax.plot(cnt[13:],linewidth=2)
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
plt.show()

sort = True
nrecs =  np.zeros(nsxw,dtype='float32')
shots = []; recx = [];
# Get receivers for each shot
for isx in range(nsxw):
  sidxs = bsx == srcxw[isx]
  # Get receiver coords
  recs = np.asarray(recsall[sidxs])
  nrecs[isx] = len(recs)
  isht = np.asarray(datr[sidxs,:])
  if(sort):
    idxs = np.argsort(recs)
    recs = recs[idxs]
    isht = isht[idxs,:]
  recx.append(recs)
  # Apply a mute to the shot
  mut   = np.squeeze(mute(isht,dt=dt,dx=dsx,v0=5.2,t0=0.2,half=False))
  shots.append(isht)

shtsout = np.concatenate(shots,axis=0)
recxs = np.concatenate(recx,axis=0)

sep.write_file("hale_shotflatsort.H",shtsout.T,ds=[dt,1.0])
sep.write_file("hale_srcxflat.H",srcxw)
sep.write_file("hale_recxflatsort.H",recxs)
sep.write_file("hale_nrec.H",nrecs)


