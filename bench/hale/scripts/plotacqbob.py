import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in coordinates
daxes,dat= sep.read_file("stackwind.H")
[nt,nrx,nsx] = daxes.n; [ot,orx,osx] = daxes.o; [dt,drx,dsx] = daxes.d
dat = dat.reshape(daxes.n,order='F').T
vmin = np.min(dat); vmax= np.max(dat); sc = 0.3

# Create shot and receiver coordinates
srcx = np.linspace(osx,osx+(nsx-1)*dsx,nsx)
recx = np.linspace(orx,orx+(nrx-1)*drx,nrx)

# Read in the velocity model
vaxes,vel = sep.read_file("vintz.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

zplt = dz*10
for isx in range(nsx):
  # Get source and receiver for the current shot
  isrcx = srcx[isx];
  irecx = isrcx + recx[:]
  zeros = np.zeros(len(irecx)) + zplt
  fig1 = plt.figure(figsize=(14,7)); ax1 = fig1.gca()
  ax1.imshow(velin[:,0,:],cmap='jet',extent=[ovx,ovx+nvx*dvx,nz*dz,0])
  ax1.set_xlabel('X (km)',fontsize=15)
  ax1.set_ylabel('Z (km)',fontsize=15)
  ax1.tick_params(labelsize=15)
  print("bob")
  print(isrcx)
  print(irecx)
  ax1.scatter(isrcx,zplt,c='tab:red',marker='*')
  ax1.scatter(irecx,zeros,c='tab:green',marker='v')
  fig2 = plt.figure(figsize=(5,10)); ax2 = fig2.gca()
  ax2.imshow(dat[isx].T,cmap='gray',vmin=vmin*sc,vmax=vmax*sc,extent=[orx,orx+(nrx-1)*drx,nt*dt,0],
             aspect=1.5)
  ax2.set_title('Srcx=%.3f'%(isrcx),fontsize=15)
  ax2.set_xlabel('X (km)',fontsize=15)
  ax2.set_ylabel('Time (s)',fontsize=15)
  ax2.tick_params(labelsize=15)
  plt.show()

