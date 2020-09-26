import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep()

daxes,dat = sep.read_file("intro_dat.H")
dat = dat.reshape(daxes.n,order='F').T
[nt,nrx,nsx] = daxes.n; [dt,drx,dsx] = daxes.d
drx /= 1000.0

sc = 0.01
vmin = sc*np.min(dat); vmax = sc*np.max(dat)

#fsize = 15
#for isht in range(nsx):
#  fig = plt.figure(figsize=(5,10)); ax = fig.gca()
#  ax.imshow(dat[isht].T,cmap='gray',interpolation='sinc',
#            extent=[0,nrx*drx,nt*dt,0],vmin=vmin,vmax=vmax,aspect=4.0)
#  ax.set_ylabel('Time (s)',fontsize=fsize)
#  ax.set_xlabel('X (km)',fontsize=fsize)
#  ax.tick_params(labelsize=fsize)
#  plt.savefig('./fig/shts/sht%d.png'%(isht),bbox_inches='tight',dpi=150,transparent=True)

