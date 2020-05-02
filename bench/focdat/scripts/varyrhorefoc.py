import inpout.seppy as seppy
import numpy as np
from resfoc.estro import refocusimg
from resfoc.gain import agc
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in images
iaxes,img = sep.read_file("../image/fltimgbigresangstkwrng.H")
img = img.reshape(iaxes.n,order='F')
img = np.ascontiguousarray(img.T).astype('float32')

[nz,nx,nro] = iaxes.n; [dz,dx,dro] = iaxes.d;

# Read in semblance rho
raxes,rsmb = sep.read_file("../image/rhofitnormwrng.H")
rsmb = rsmb.reshape(raxes.n,order='F')
rsmb = np.ascontiguousarray(rsmb.T).astype('float32')

# Read in image space rho
raxes,rimg = sep.read_file("../image/rhotgtimg.H")
rimg = rimg.reshape(raxes.n,order='F')
rimg = np.ascontiguousarray(rimg.T).astype('float32')

# Refocusing
rfis = refocusimg(img,rsmb,dro)

# Write the refocused images
sep.write_file("refocsemb.H",rfis.T,ds=[dz,dx])

rfii = refocusimg(img,rimg,dro)
sep.write_file("refocimage.H",rfii.T,ds=[dz,dx])

# Plotting
vmin = np.min(agc(img[16])); vmax = np.max(agc(img[16]))
pclip = 0.5
fsize = 15
wbox = 10; hbox = 6

fig1 = plt.figure(1,figsize=(wbox,hbox)); ax1 = fig1.gca()
ax1.imshow(agc(rfis).T,cmap='gray',interpolation='sinc',extent=[0.0,nx*dx/1000.0,(nz)*dz/1000.0,0.0],
           vmin=pclip*vmin,vmax=pclip*vmax)
ax1.set_xlabel('X (km)',fontsize=fsize)
ax1.set_ylabel('Z (km)',fontsize=fsize)
ax1.tick_params(labelsize=fsize)
ax1.set_title('Semblance',fontsize=fsize)
plt.savefig('./fig/refocsemb.png',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

fig2 = plt.figure(2,figsize=(wbox,hbox)); ax2 = fig2.gca()
ax2.imshow(agc(rfii).T,cmap='gray',interpolation='sinc',extent=[0.0,nx*dx/1000.0,(nz)*dz/1000.0,0.0],
           vmin=pclip*vmin,vmax=pclip*vmax)
ax2.set_xlabel('X (km)',fontsize=fsize)
ax2.set_ylabel('Z (km)',fontsize=fsize)
ax2.tick_params(labelsize=fsize)
ax2.set_title('Image space',fontsize=fsize)
plt.savefig('./fig/refocimage.png',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

fig3 = plt.figure(3,figsize=(wbox,hbox)); ax3 = fig3.gca()
ax3.imshow(agc(img[16]).T,cmap='gray',interpolation='sinc',extent=[0.0,nx*dx/1000.0,(nz)*dz/1000.0,0.0],
           vmin=pclip*vmin,vmax=pclip*vmax)
ax3.set_xlabel('X (km)',fontsize=fsize)
ax3.set_ylabel('Z (km)',fontsize=fsize)
ax3.tick_params(labelsize=fsize)
ax3.set_title('Defocused',fontsize=fsize)
plt.savefig('./fig/defocused.png',dpi=150,transparent=True,bbox_inches='tight')
plt.close()

#plt.show()
