import inpout.seppy as seppy
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

begz = 0.3; begx = 0.5

dfz = 5000.0; daz = 5000.0
zbeg = begz*zdist; xbeg = begx*xdist

theta0 = np.arctan2(dfz, daz);
theta0 = theta0 + 2 * np.pi;
theta0 = theta0 * 180. / np.pi;
print(theta0-360)

zcenter = zbeg - dfz
xcenter = xbeg - daz

fullradius = np.sqrt(dfz*dfz + daz*daz)

shiftx = np.zeros([nx,nz])
shiftz = np.zeros([nx,nz])

distdie = 0.3
distdie *= 5000
thetashift = 1.0
thetadie=12.0

traj = np.zeros([nx,nz])

# Compute shifts
for ix in range(nx):
  px = dx * ix - xcenter

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

    if(np.abs(radius - fullradius) < 50):
      traj[ix,iz] = 1.0

    # Check if we are in the region for faulting
    # Criteria for radius and angle
    ratioAz    = np.abs(fullradius - radius)/distdie
    ratioTheta = np.abs(thetaCompare - theta0)/thetadie
 
    # Make sure not too far away from xbeg
    diffx = xbeg - (px + xcenter)
    diffz = zbeg - (pz + zcenter)
    distbeg = np.sqrt(diffx*diffx + diffz*diffz)

    if(ratioAz < 1 and ratioTheta < 1 and distbeg < 2000):
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
      newX = radius * np.cos(thetaNew * np.pi/180) + xcenter
      newZ = radius * np.sin(thetaNew * np.pi/180) + zcenter

      shiftz[ix,iz] = (newZ - dz*iz)
      shiftx[ix,iz] = (newX - dx*ix)
 

sep = seppy.sep()
sep.write_file("shiftz.H",shiftz.T,ds=[dz,dx])
sep.write_file("shiftx.H",shiftx.T,ds=[dz,dx])
sep.write_file("layers.H",tot.T,ds=[dz,dx])

