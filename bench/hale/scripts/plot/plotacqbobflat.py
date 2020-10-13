import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in coordinates
saxes,srcx = sep.read_file("hale_srcxflatbob.H")
raxes,recx = sep.read_file("hale_recxflatbob.H")
_,nrec= sep.read_file("hale_nrecbob.H")
nrec = nrec.astype('int')

# Read in the velocity model
vaxes,vel = sep.read_file("vintz.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

# Read in the data
daxes,dat = sep.read_file("hale_shotflatbob.H")
dat = dat.reshape(daxes.n,order='F').T
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d
vmin = np.min(dat); vmax= np.max(dat); sc = 0.3

beg = 0; end = nrec[0]
zplt = dz*10
for isx in range(len(srcx)-1):
  # Get source and receiver for the current shot
  isrcx = srcx[isx];
  irecx = recx[beg:end]; zeros = np.zeros(nrec[isx]) + zplt
  zeros = np.zeros(len(irecx)) + zplt
  fig1 = plt.figure(figsize=(14,7)); ax1 = fig1.gca()
  ax1.imshow(velin[:,0,:],cmap='jet',extent=[ovx,ovx+nvx*dvx,nz*dz,0])
  ax1.set_xlabel('X (km)',fontsize=15)
  ax1.set_ylabel('Z (km)',fontsize=15)
  ax1.tick_params(labelsize=15)
  ax1.scatter(isrcx,zplt,c='tab:red',marker='*')
  ax1.scatter(irecx,zeros,c='tab:green',marker='v')
  fig2 = plt.figure(figsize=(5,10)); ax2 = fig2.gca()
  rxl = irecx[-1] - irecx[0]
  ax2.imshow(dat[beg:end,:].T,cmap='gray',vmin=vmin*sc,vmax=vmax*sc,extent=[0,rxl,nt*dt,0],
             aspect=1.5)
  ax2.set_title('Srcx=%.3f'%(isrcx),fontsize=15)
  ax2.set_xlabel('X (km)',fontsize=15)
  ax2.set_ylabel('Time (s)',fontsize=15)
  ax2.tick_params(labelsize=15)
  plt.show()
  # Update positions
  beg = end; end += nrec[isx+1]

