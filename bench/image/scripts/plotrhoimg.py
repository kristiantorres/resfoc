import inpout.seppy as seppy
import numpy as np
from scaas.trismooth import smooth
from resfoc.estro import estro_tgt
from resfoc.gain import agc
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in stack
#saxes,stk = sep.read_file("fltimgbigresangstkwrng.H") # Angle stack
saxes,stk = sep.read_file("fltimgbigreswrng.H")        # Zero subsurface offset
stk = stk.reshape(saxes.n,order='F')
stk = np.ascontiguousarray(stk.T).astype('float32')
stk = stk[:,16,:,:]

#[nz,nx,nro] = saxes.n; [oz,ox,oro] = saxes.o; [dz,dx,dro] = saxes.d
[nz,nx,nh,nro] = saxes.n; [oz,ox,oh,oro] = saxes.o; [dz,dx,dh,dro] = saxes.d

#viewimgframeskey(agc(stk),transp=True,pclip=0.8,interp='sinc',show=False,
#    ottl=oro,dttl=dro,ttlstring=r'$\rho$=%.3f')

# Read in well-focused image
#faxes,foc = sep.read_file("fltimgbig.H")         # Zero subsurface offset
faxes,foc = sep.read_file("fltimgbigresangstk.H") # Residually migrated Angle stack
foc = foc.reshape(faxes.n,order='F')
foc = np.ascontiguousarray(foc.T).astype('float32')

zofoc = foc[16,:,:]

# Read in rho picks
paxes,pck = sep.read_file("rhofitnormwrng.H")
pck = pck.reshape(paxes.n,order='F')
pck = np.ascontiguousarray(pck.T).astype('float32')

# Compute image space rho
# Pad the stack and the zero subsurface offset
stkp = np.pad(stk,((0,0),(12,12),(0,0)),'edge')
zofocp = np.pad(zofoc,((12,12),(0,0)),'edge')
rhoi = estro_tgt(stkp,zofocp,dro,oro,nzp=64,nxp=64,strdx=32,strdz=32)
rhoism = smooth(rhoi.astype('float32'),rect1=30,rect2=30)
# Window rhoiosm
rhoismwind = rhoism[11:11+1000,:]
sep.write_file("rhotgtimg.H",rhoismwind.T,ds=[dz,dx])

vmin = np.min(agc(stk[16,:,:])); vmax = np.max(agc(stk[16,:,:]))

# Plot results
fsize = 15; pclip = 0.4
fig1 = plt.figure(2,figsize=(10,6))
ax1 = fig1.gca()
ax1.imshow(agc(stk[16,:,:]).T,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=vmin*pclip,vmax=vmax*pclip)
im1 = ax1.imshow(rhoismwind.T,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',
                 vmin=0.98,vmax=1.02,alpha=0.1)
ax1.set_xlabel('X (km)',fontsize=fsize)
ax1.set_ylabel('Z (km)',fontsize=fsize)
ax1.tick_params(labelsize=fsize)
ax1.set_title(r"Unfocused $h_0$ and Focused Stack",fontsize=fsize)
#ax1.set_title(r"Unfocused $h_0$ and Focused $h_0$",fontsize=fsize)
#ax1.set_title(r"Unfocused Stack and Focused $h_0$",fontsize=fsize)
#ax1.set_title(r"Unfocused Stack and Focused Stack",fontsize=fsize)
cbar_ax1 = fig1.add_axes([0.91,0.15,0.02,0.70])
cbar1 = fig1.colorbar(im1,cbar_ax1,format='%.2f')
cbar1.solids.set(alpha=1)
cbar1.ax.tick_params(labelsize=fsize)
cbar1.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/rhoimgh0stack.png',bbox_inches='tight',transparent=True,dpi=150)
plt.close()

fig2 = plt.figure(3,figsize=(10,6))
ax2 = fig2.gca()
ax2.imshow(agc(stk[16,:,:]).T,cmap='gray',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='sinc',vmin=vmin*pclip,vmax=vmax*pclip)
im2 = ax2.imshow(pck.T,cmap='seismic',extent=[0.0,(nx)*dx/1000.0,nz*dz/1000.0,0.0],interpolation='bilinear',vmin=0.98,vmax=1.02,alpha=0.1)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
ax2.set_title(r"Semblance",fontsize=fsize)
cbar_ax2 = fig2.add_axes([0.91,0.15,0.02,0.70])
cbar2 = fig2.colorbar(im2,cbar_ax2,format='%.2f')
cbar2.solids.set(alpha=1)
cbar2.ax.tick_params(labelsize=fsize)
cbar2.set_label(r'$\rho$',fontsize=fsize)
plt.savefig('./fig/rhogat.png',bbox_inches='tight',transparent=True,dpi=150)
plt.close()

#plt.show()
