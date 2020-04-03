import inpout.seppy as seppy
import resfoc.cosftsimp as cfts
import matplotlib.pyplot as plt

# Read in the data
sep = seppy.sep([])
paxes,pln = sep.read_file('plane.H',form='native')
pln = pln.reshape(paxes.n,order='F')

plnft1  = cfts.cosft(pln,axis1=1)
plnft2  = cfts.cosft(pln,axis2=1)
plnft12 = cfts.cosft(pln,axis1=1,axis2=1)

iplnft1  = cfts.icosft(plnft1,axis1=1)
iplnft2  = cfts.icosft(plnft2,axis2=1)
iplnft12 = cfts.icosft(plnft12,axis1=1,axis2=1)

# Plot results
plt.figure()
plt.imshow(pln)

f1,ax1 = plt.subplots(1,3,figsize=(10,5))
ax1[0].imshow(plnft1)
ax1[1].imshow(plnft2)
ax1[2].imshow(plnft12)

f2,ax2 = plt.subplots(1,3,figsize=(10,5))
ax2[0].imshow(iplnft1)
ax2[1].imshow(iplnft2)
ax2[2].imshow(iplnft12)
plt.show()


