import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey

sep = seppy.sep()

# Midpoints
maxes,mid = sep.read_file("./dat/mymidpts.H")

[nt,nh,nm] = maxes.n; [dt,dh,dm] = maxes.d; [ot,oh,om] = maxes.o

mid = mid.reshape(maxes.n,order='F').T

viewimgframeskey(mid,pclip=0.2,interp='none',zmax=nt*dt,xmin=oh,xmax=oh+nh*dh,hbox=10,wbox=4)

# Semblance
saxes,smb = sep.read_file("myscn.H")

[nt,nv,nm] = saxes.n; [dt,dv,dm] = saxes.d; [ot,ov,om] = saxes.o

smb = np.ascontiguousarray(smb.reshape(saxes.n,order='F').T).astype('float32')

smbsm = smooth(smb,rect1=20)

viewimgframeskey(smbsm,cmap='jet',interp='bilinear',zmax=nt*dt,xmin=ov,xmax=ov+nv*dv,hbox=10,wbox=4)

# Plot rms velocity on one of them
vaxes,vel = sep.read_file("velrms.H")
vel = vel.reshape(vaxes.n,order='F').T

idx = 125
t = np.linspace(0,nt*dt,nt)
fig = plt.figure(figsize=(5,10)); ax = fig.gca()
ax.imshow(smbsm[idx].T,cmap='jet',interpolation='bilinear',extent=[ov,ov+nv*dv,nt*dt,0],aspect=0.8)
ax.tick_params(labelsize=16)
ax.plot(vel[idx],t,linewidth=3,color='k')
plt.show()

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(vel.T,cmap='jet',interpolation='bilinear',extent=[om,om+nm*dm,nt*dt,0])
ax.tick_params(labelsize=16)
ax.set_title('Vrms',fontsize=16)

# Read in stack
naxes,nmo = sep.read_file("stk.H")
nmo = nmo.reshape(naxes.n,order='F').T

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(nmo.T,cmap='gray',interpolation='sinc',extent=[om,om+nm*dm,nt*dt,0],
         vmin=-2e6,vmax=2e6)
ax.tick_params(labelsize=16)
ax.set_title('Stack',fontsize=16)
plt.show()


# Read in two interval velocities
ivaxes,ivel = sep.read_file("vint.H")
ivel = ivel.reshape(ivaxes.n,order='F').T

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(ivel.T,cmap='jet',interpolation='bilinear',extent=[om,om+nm*dm,nt*dt,0])
ax.tick_params(labelsize=16)
ax.set_ylabel('Time (s)',fontsize=16)
ax.set_title('Vint(t)',fontsize=16)

ivaxes,ivel = sep.read_file("vintz.H")
[nz,_] = ivaxes.n; [dz,_] = ivaxes.d
ivel = ivel.reshape(ivaxes.n,order='F').T

fig = plt.figure(figsize=(10,10)); ax = fig.gca()
ax.imshow(ivel.T,cmap='jet',interpolation='bilinear',extent=[om,om+nm*dm,nz*dz,0])
ax.tick_params(labelsize=16)
ax.set_ylabel('Z (km)',fontsize=16)
ax.set_title('Vint(z)',fontsize=16)


plt.show()


