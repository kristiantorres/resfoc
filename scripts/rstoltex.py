import inpout.seppy as seppy
import numpy as np
import resfoc.cosft as cft
import resfoc.rstolt as rstolt
import matplotlib.pyplot as plt

# Set up IO
sep = seppy.sep([])

# Read in the image
iaxes,img = sep.read_file(None,ifname="extpt.H")
img = img.reshape(iaxes.n,order='F')

nz = iaxes.n[0]; nx = iaxes.n[1]; nh = iaxes.n[2]
dz = iaxes.d[0]; dx = iaxes.d[1]; dh = iaxes.d[2]
oh = iaxes.o[2]

# Pad the image in each direction
pimg = np.pad(img,((0,513-nz),(0,513-nx),(0,33-nh)),'constant')

pimgft = cft.cosft(pimg,axis1=1,axis2=1,axis3=1)
dcs = cft.samplings(pimgft,iaxes)
#print(dcs)

#print(pimg.shape)
#plt.imshow(pimg[:,:,10],cmap='gray'); plt.show()

#oaxes = seppy.axes(pimgft.shape,[0.0,0.0,0.0],dcs)
#sep.write_file(None,oaxes,pimgft,ofname='mystoltft.H')

print(pimgft.shape)
pimgftt = np.ascontiguousarray(np.transpose(pimgft,(2,1,0)))
pimgftt = pimgftt.astype('float32')
print(pimgftt.shape)
print(pimgftt.flags)
nzp = pimgftt.shape[2]; nmp = pimgftt.shape[1]; nhp = pimgftt.shape[0]; nro = 2
print(nzp,nmp,nhp,nro)
rst = rstolt.rstolt(nzp,nmp,nhp,nro,dcs[0],dcs[1],dcs[2],0.01,1.0)

rmig = np.zeros([2*nro-1,nhp,nmp,nzp],dtype='float32')
rst.resmig(pimgftt,rmig,2)
rmigt = np.transpose(rmig,(3,2,1,0))
raxes = seppy.axes([nzp,nmp,nhp,2*nro-1],[0.0,0.0,0.0,98],[dcs[0],dcs[1],dcs[2],0.01])
sep.write_file(None,raxes,rmigt,ofname='inrstolt.H')

# Inverse cosine transform
rmigift = cft.icosft(rmigt,axis1=1,axis2=1,axis3=1)

# Remove the padding
rmigiftwind  = rmigift[0:nz,0:nx,0:nh,:]
nraxes = seppy.axes([nz,nx,nh,2*nro-1],[0.0,0.0,oh,0.97],[dz,dx,dh,0.01])
sep.write_file(None,nraxes,rmigiftwind,ofname='iniftwind.H')

