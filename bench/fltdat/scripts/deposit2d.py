import numpy as np
import matplotlib.pyplot as plt

def deposit(mod,ibot,itop,dz,vel,layerT=10,dev_layer=0.1):
  if(itop > ibot):
    raise Exception("Top sample must be less than bottom sample")
  # Create the layering function
  vu = 1. + ((np.random.rand() - .5) * dev_layer)
  nz = np.abs(itop-ibot)
  layv = np.zeros(nz)
  iold = 0
  for iz in range(nz):
    ii = int((iz - iold) * dz / layerT)
    if(ii != iold):
      vu = 1. + ((np.random.rand() - .5) * dev_layer)
    iold = ii
    layv[iz] = vu
  # Add it to the velocity model
  for ix in range(nx):
    mod[ix,itop+1:ibot+1] = vel*layv


nx = 1000; nz = 1000
nlyr = 2
tot = np.zeros([nx,nz])

dlyr = 50; bot = 999
minvel = 1600; maxvel = 3000
props = np.linspace(maxvel,minvel,19)
dz = 12.5
for ilyr in range(len(props)):
  deposit(tot,bot,bot-dlyr,dz=dz,vel=props[ilyr],layerT=100,dev_layer=0.05)
  bot -= dlyr

widx = tot == 0
tot[widx] = 1500.0

#plt.imshow(tot.T,cmap='jet')
#plt.show()

# Put in a fault
dx = 25
xdist = dx*nx
zdist = dz*nz

begz = 0.9; begx = 0.8

dfz = 5000.0; daz = 5000.0
zbeg = begz*zdist; xbeg = begx*xdist

theta0 = np.arctan2(dfz, daz);
theta0 = theta0 + 2 * np.pi;
theta0 = theta0 * 180. / np.pi;
print(theta0-360)

azim = 0.0
azim *= np.pi/180.0
azicor = np.cos(azim)
zcenter = zbeg - dfz
xcenter = xbeg - daz*azicor

fullradius = np.sqrt(dfz*dfz + daz*daz)

shiftx = np.zeros([nx,nz])
shiftz = np.zeros([nx,nz])

distdie = 0.3
distdie *= 5000
thetashift = 1.0
thetadie=12.0

circ = np.zeros([nx,nz])

# Compute shifts
for ix in range(nx):
  px = dx * ix - xcenter
  px *= azicor

  for iz in range(nz):
    pz = dz * iz - zcenter

    # Compute the angle from vertical for the current point
    thetaOld = np.arctan2(pz,px) * 180 / np.pi + 360
    if(pz < 0 and px == 0.0):
      thetaCompare = 270
    elif(pz > 0 and px == 0.0):
      thetaCompare = 450
    elif(pz == 0 and px == 0.0):
      thetaCompare = 100000
    else:
      thetaCompare = np.arctan(pz/px) * 180 / np.pi + 360

    # Compute the distance from center for the current point
    radius = np.sqrt(px*px + pz*pz)

    if(np.abs(radius - fullradius) < 20):
      circ[ix,iz] = 1.0 

    # Check if we are in the region for faulting
    # Criteria for radius and angle
    ratioAz    = np.abs(fullradius - radius)/distdie
    ratioTheta = np.abs(thetaOld - theta0)/thetadie
 
    # Make sure not too far away from xbeg
    diffx = xbeg - (px*azicor + xcenter)
    diffz = zbeg - (pz        + zcenter)
    distbeg = np.sqrt(diffx*diffx + diffz*diffz)

    if(ratioAz < 1 and ratioTheta < 1):
      print(thetaCompare,theta0)
    #if(ratioAz < 1 and ratioTheta < 1):
      # Once we are in range, compute the displacement
      scaleAz    = 1 - ratioAz
      scaleTheta = 1 - ratioTheta

      # Problem, thetashift controls the throw and scaleaz and scaletheta
      # control the distance. This means that only large faults will have large throw
      shifttheta = thetashift * scaleAz * scaleTheta

      thetaNew = thetaOld + shifttheta
      # If outside the circle, flip the sign of the shift
      if(radius > fullradius):
        thetaNew = thetaOld - shifttheta

      # Convert back to cartesian coordinates 
      newX = radius * np.cos(thetaNew * np.pi/180)*azicor + xcenter
      newZ = radius * np.sin(thetaNew * np.pi/180)        + zcenter

      shiftz[ix,iz] = (newZ - dz*iz)
      shiftx[ix,iz] = (newX - dx*ix)
      circ[ix,iz] *= shiftz[ix,iz]

#print(xcenter,zcenter)
#print(fullradius)
## Draw circle
#circ = np.zeros([nx,nz])
#for ix in range(nx):
#  x = ix*dx - xcenter
#  for iz in range(nz):
#    z = iz*dz - zcenter
#    pr = np.sqrt(x*x + z*z)
#    if(np.abs(pr - fullradius) < 20):
#      circ[ix,iz] = 1.0 

plt.figure(1)
plt.imshow(shiftx.T,cmap='jet',extent=[0,nx*dx,nz*dz,0])
plt.scatter(xcenter,zcenter)
plt.scatter(xbeg,zbeg)
plt.figure(2)
plt.imshow(shiftz.T,cmap='jet',extent=[0,nx*dx,nz*dz,0])
plt.scatter(xcenter,zcenter)
plt.scatter(xbeg,zbeg)
plt.figure(3)
plt.imshow(circ.T,cmap='jet',extent=[0,nx*dx,nz*dz,0])
plt.scatter(xcenter,zcenter)
plt.scatter(xbeg,zbeg)
plt.figure(4)
plt.imshow((circ*shiftz).T,cmap='jet',extent=[0,nx*dx,nz*dz,0])
plt.show()

velflt = np.zeros([nx,nz])

# Apply shifts
for ix in range(nx):
  for iz in range(nz):
    l1 = max(0,int((iz-shiftz[ix,iz]/dz + 0.5)))
    l2 = max(0,int((ix-shiftx[ix,iz]/dx + 0.5)))
    if(l1 >= nz): l1 = nz - 1
    if(l2 >= nx): l2 = nx - 1
    if(l1 >= 0):
      velflt[ix,iz] = tot[l2,l1]

plt.imshow(velflt.T,cmap='jet')
plt.show()

