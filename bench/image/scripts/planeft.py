import inpout.seppy as seppy
import resfoc.cosft as cft
import matplotlib.pyplot as plt

# Read in the data
sep = seppy.sep()
paxes,pln = sep.read_file('plane.H',form='native')
pln = pln.reshape(paxes.n,order='F').astype('float32')

plnft1  = cft.cosft(pln.T,axis1=1)
plnft2  = cft.cosft(pln.T,axis0=1)
plnft12 = cft.cosft(pln.T,axis0=1,axis1=1)

iplnft1  = cft.icosft(plnft1,axis1=1)
iplnft2  = cft.icosft(plnft2,axis0=1)
iplnft12 = cft.icosft(plnft12,axis0=1,axis1=1)

## Plot results
plt.figure()
plt.imshow(pln)

f1,ax1 = plt.subplots(1,3,figsize=(10,5))
ax1[0].imshow(plnft1.T)
ax1[1].imshow(plnft2.T)
ax1[2].imshow(plnft12.T)

f2,ax2 = plt.subplots(1,3,figsize=(10,5))
ax2[0].imshow(iplnft1.T)
ax2[1].imshow(iplnft2.T)
ax2[2].imshow(iplnft12.T)
plt.show()
#
sep.write_file("planeft.H",plnft1.T)
