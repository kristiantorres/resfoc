import numpy as np
import resfoc.cosft as cft
import matplotlib.pyplot as plt

def samplings(nsin,dsin):
  """ Computes the cosine transformed samplings """
  ns = nsin
  ndim = len(ns)
  ds = []
  for idim in range(ndim):
    ds.append(1/(2*cft.next_fast_size(ns[idim]-1)*dsin[idim]))

  return ds

nz = 513; nx = 513; nh = 2049;
dz = 10;  dx = 10; dh = 10
oz = 0;   ox = 0; oh = -160.0

dcs = samplings([nh,nx,nz],[dh,dx,dz])

dkh = np.pi*dcs[0]; dkx = np.pi*dcs[1]; dkz = np.pi*dcs[2]

#print(dkh,dkx,dkz)

ih = 16
kh = ih*dkh
stretches = np.zeros([nx,nz])
stretch = np.zeros(nz)
kzs = np.zeros([nx,nz])

for ix in range(nx):
  km = ix*dkx
  for iz in range(1,nz):
    kz = iz*dkz; kzs[ix,iz] = kz
    kzh = kz*kz + kh*kh
    kzm = kz*kz + km*km
    zzs = (kzh*kzm) - (kz*kz) * ( (km-kh)*(km-kh) )
    zzg = (kzh*kzm) - (kz*kz) * ( (km+kh)*(km+kh) )
    if(zzs > 0 and zzg > 0):
      stretch[iz] = 0.5/kz * ( np.sqrt(zzs) + np.sqrt(zzg) )
    else:
      stretch[iz] = -2.0 * dkz
    #print("im=%d iz=%d kh=%f km=%f str[iz]=%f kz=%f"%(ix,iz,kh,km,stretch[iz],kz))
  stretches[ix,:] = stretch[:]

plt.figure()
plt.imshow(stretches.T,vmin=np.min(kzs),vmax=np.max(kzs))
plt.title(r"$k_z$")
plt.figure()
plt.imshow(kzs.T,vmin=np.min(kzs),vmax=np.max(kzs))
plt.title(r"$k_{z_0}$")
plt.figure()
plt.imshow((stretches-kzs).T,vmin=np.min(kzs),vmax=np.max(kzs))
plt.title("Difference")
plt.show()

