import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
import matplotlib.pyplot as plt

sep = seppy.sep()
aaxes,ang = sep.read_file('x600wrng.H')
ang = ang.reshape(aaxes.n,order='F')
ang = np.ascontiguousarray(ang.T).astype('float32')

[nz,na,nro] = aaxes.n; [oz,oa,oro] = aaxes.o; [dz,da,dro] = aaxes.d

stack = np.sum(ang,axis=1)

# Smoothing paramters
rect1=10; rect2=3

# Numerator
stacksq = stack*stack
num = smooth(stacksq.astype('float32'),rect1=rect1,rect2=rect2)

# Denominator
sqstack = np.sum(ang*ang,axis=1)
denom = smooth(sqstack.astype('float32'),rect1=rect1,rect2=rect2)

# Semblance
semb = num/denom
sep.write_file("x600sembwrng.H",semb.T,os=[oz,oro],ds=[dz,dro])

# Plotting parameters
wbox = 10; hbox=6
fsize = 15

# Plot the semblance
fig = plt.figure(1,figsize=(wbox,hbox))
ax = fig.gca()
ax.imshow(semb.T,cmap='jet',aspect=0.02,extent=[oro,oro+(nro)*dro,nz*dz/1000.0,0.0],interpolation='bilinear')
ax.set_xlabel(r'$\rho$',fontsize=fsize)
ax.set_ylabel('Z (km)',fontsize=fsize)
ax.tick_params(labelsize=fsize)

# Plot the data as a function of rho
angr = ang.reshape([na*nro,nz])

vmin = np.min(angr); vmax = np.max(angr); pclip=0.9

fig2 = plt.figure(2,figsize=(wbox,hbox))
ax2 = fig2.gca()
ax2.imshow(angr.T,cmap='gray',aspect=0.009,extent=[oro,oro+(nro)*dro,nz*dz/1000.0,0.0],interpolation='sinc',
           vmin=vmin*pclip,vmax=vmax*pclip)
ax2.set_xlabel(r'$\rho$',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
plt.show()

