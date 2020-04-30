import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
import matplotlib.pyplot as plt

# Read in data
sep = seppy.sep()
aaxes,ang = sep.read_file('x500.H')
ang = ang.reshape(aaxes.n,order='F')
ang = np.ascontiguousarray(ang.T).astype('float32')

[nz,na,nro] = aaxes.n; [oz,oa,oro] = aaxes.o; [dz,da,dro] = aaxes.d

print(oro,dro)

# Read in semblance
saxes,semb = sep.read_file("x500semb.H")
semb = semb.reshape(saxes.n,order='F')
semb = np.ascontiguousarray(semb.T).astype('float32')

# Read in pick
paxes,pick = sep.read_file("x500pick.H")
pick = pick.reshape(paxes.n,order='F')
pick = np.ascontiguousarray(pick.T).astype('float32')

# Plot the results

# Plotting parameters
wbox = 10; hbox=6
fsize = 15

z = np.linspace(0.0,(nz-1)*dz/1000.0,nz)

# Plot the semblance
fig = plt.figure(1,figsize=(wbox,hbox))
ax = fig.gca()
ax.imshow(semb.T,cmap='jet',aspect=0.02,extent=[oro,oro+(nro)*dro,nz*dz/1000.0,0.0],interpolation='bilinear')
ax.plot(pick,z,linewidth=3,color='k')
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
ax2.plot(pick,z,linewidth=3,color='tab:cyan')
ax2.set_xlabel(r'$\rho$',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
plt.show()


