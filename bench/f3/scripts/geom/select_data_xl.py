import inpout.seppy as seppy
import segyio
from inpout.seppy import bytes2float
import numpy as np
from genutils.ptyprint import printprogress
from genutils.plot import plot_img2d
from genutils.movie import viewcube3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Grid origin and sampling of the original grid
ox,oy = 469800,6072350
dz,dx,dy = 10,25,25

# Data sampling
dt = 0.002

sep = seppy.sep()
# Read in the migration cube
maxes,mig = sep.read_file("./mig/mig.T")
mig = mig.reshape(maxes.n,order='F')

# Window the migration cube
migw = mig[:,200:1200,5:505]

# Read in the velocity model
vaxes,vel = sep.read_file("./vels/miglintz.H")
vel = vel.reshape(vaxes.n,order='F')

# Windowed grid (based on velocity model windowing)
oxw = ox + 200*dx; oyw = oy + 5*dy
nzw1,nxw1,nyw1 = vel.shape

# Plotting axes
oxp,oyp = oxw*0.001,oyw*0.001
dzp,dxp,dyp = dz*0.001,dx*0.001,dy*0.001

# Second round of windowing
#TODO: need a better way of specifying the bounding box then
#      just changing these parameters
velw  = vel[:400,:500,:500]
migww = migw[:800,:500,:500]
nzw2,nxw2,nyw2 = velw.shape

# Plot the region of interest on the full migration cube and velocity model
migslc = migw[400]
#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#im = ax.imshow(np.flipud(migslc.T),extent=[oxp,oxp+nxw*dxp,oyp,oyp+nyw*dyp],interpolation='bilinear',cmap='gray')
#rect = patches.Rectangle((oxp,oyp),500*dxp,50*dyp,linewidth=2,edgecolor='yellow',facecolor='none')
#ax.add_patch(rect)
#
#velslc = velw[200]
#fig = plt.figure(figsize=(10,5)); ax = fig.gca()
#im = ax.imshow(np.flipud(velslc.T),extent=[oxp,oxp+nxw*dxp,oyp,oyp+nyw*dyp],interpolation='bilinear',cmap='jet')
#rect = patches.Rectangle((oxp,oyp),500*dxp,50*dyp,linewidth=2,edgecolor='yellow',facecolor='none')
#ax.add_patch(rect)
#plt.show()

# Read in the source hash map and the source coordinates
hmap = np.load('./segy/info/scoordhmap.npy',allow_pickle=True)[()]
crds = np.load('./segy/info/scoords.npy',allow_pickle=True)[()]

# Make a 100X100m grid and loop over each point
dsx,dsy = 100,100
oyw += dy*0; oxw += dx*0 # Change coefficients of dy and dx to shift box
sxs = np.arange(oxw,oxw+(nxw2-1)*dx,dsx)
sys = np.arange(oyw,oyw+(nyw2-1)*dy,dsy)
nsx = len(sxs)
nsy = len(sys)

fcrds = []
# Remove the shots outside of the region
for icrd in range(len(crds)):
  if(crds[icrd,0] >= oyw and crds[icrd,1] >= oxw):
    fcrds.append(crds[icrd])
fcrds = np.asarray(fcrds)

ctr = 0; qc = False
dmap = {}
rsxs,rsys = [],[]
for isy in range(nsy):
  gsy = sys[isy]
  for isx in range(nsx):
    printprogress("nshots:",ctr,nsy*nsx)
    gsx = sxs[isx]
    # Compute the distance between this point and all of the source coordinates
    dists = np.sqrt((gsy - fcrds[:,0])**2 + (gsx - fcrds[:,1])**2)
    # Find this coordinate
    idx = np.argmin(dists)
    rsy,rsx = fcrds[idx]
    rsys.append(rsy); rsxs.append(rsx)
    # Get the key that tells us which files contain the data
    key = str(int(rsy)) + ' ' + str(int(rsx))
    recxinfo,recyinfo = [],[]
    strminfo = []
    srcdat = []
    for ifile in hmap[key]:
      if(ifile not in dmap):
        dmap[ifile] = segyio.open(ifile,ignore_geometry=True)
      # Get the source coordinates
      srcxf = np.asarray(dmap[ifile].attributes(segyio.TraceField.SourceX),dtype='int32')
      srcyf = np.asarray(dmap[ifile].attributes(segyio.TraceField.SourceY),dtype='int32')
      # Get the receiver coordinates
      recxf = np.asarray(dmap[ifile].attributes(segyio.TraceField.GroupX),dtype='int32')
      recyf = np.asarray(dmap[ifile].attributes(segyio.TraceField.GroupY),dtype='int32')
      # Get the streamer header
      strmr = np.asarray(dmap[ifile].attributes(segyio.TraceField.CDP_TRACE),dtype='int32')
      # Find the traces with that source coordinate
      scoordsf = np.zeros([len(srcxf),2],dtype='int32')
      scoordsf[:,0] = srcyf; scoordsf[:,1] = srcxf
      idx1 = scoordsf == fcrds[idx]
      s = np.sum(idx1,axis=1)
      nidx1 = s == 2
      # Get receiver coordinates for this shot
      recxinfo.append(recxf[nidx1]); recyinfo.append(recyf[nidx1])
      # Get the streamer header for this shot
      strminfo.append(strmr[nidx1])
      # Get the data for this shot
      data = dmap[ifile].trace.raw[:]
      srcdat.append(data[nidx1,:])
    # Concatenate data from different files
    recxinfo = np.concatenate(recxinfo,axis=0)
    recyinfo = np.concatenate(recyinfo,axis=0)
    strminfo = np.concatenate(strminfo,axis=0)
    srcdat   = np.concatenate(srcdat,axis=0)
    ntr,nt = srcdat.shape
    nrec = len(recxinfo)
    if(nrec != len(recyinfo)):
      print("Warning nrecx != nrecy for shot %f %f"%(rsy,rsx))
    if(isy == 0 and isx == 0):
      sep.write_file("windowed_data/f3_srcx3_full.H",np.asarray([rsx]))
      sep.write_file("windowed_data/f3_srcy3_full.H",np.asarray([rsy]))
      sep.write_file("windowed_data/f3_recx3_full.H",recxinfo.astype('float32'))
      sep.write_file("windowed_data/f3_recy3_full.H",recyinfo.astype('float32'))
      sep.write_file("windowed_data/f3_nrec3_full.H",np.asarray([nrec],dtype='float32'))
      sep.write_file("windowed_data/f3_strm3_full.H",strminfo.astype('float32'))
      sep.write_file("windowed_data/f3_shots3_full.H",srcdat.T,ds=[dt,1.0])
    else:
      sep.append_file("windowed_data/f3_srcx3_full.H",np.asarray([rsx]))
      sep.append_file("windowed_data/f3_srcy3_full.H",np.asarray([rsy]))
      sep.append_file("windowed_data/f3_recx3_full.H",recxinfo.astype('float32'))
      sep.append_file("windowed_data/f3_recy3_full.H",recyinfo.astype('float32'))
      sep.append_file("windowed_data/f3_nrec3_full.H",np.asarray([nrec],dtype='float32'))
      sep.append_file("windowed_data/f3_strm3_full.H",strminfo.astype('float32'))
      sep.append_file("windowed_data/f3_shots3_full.H",srcdat.T)
    if(qc):
      # Plot the source receiver geometry for this shot
      fig = plt.figure(figsize=(10,5)); ax = fig.gca()
      im = ax.imshow(np.flipud(migslc.T),extent=[oxp,oxp+nxw1*dxp,oyp,oyp+nyw1*dyp],interpolation='bilinear',cmap='gray')
      ax.scatter(np.asarray(rsxs)/1000.0,np.asarray(rsys)/1000.0,marker='*',color='tab:red')
      ax.scatter(recxinfo/1000.0,recyinfo/1000.0,marker='v',color='tab:green')
      # Plot the data
      fig = plt.figure(); ax = fig.gca()
      pclip = 0.05
      dmin = pclip*np.min(srcdat); dmax = pclip*np.max(srcdat)
      ax.imshow(srcdat.T,cmap='gray',extent=[0,ntr,nt*dt,0],aspect='auto',vmin=dmin,vmax=dmax)
      ax.set_xlabel('Receiver no',fontsize=15)
      ax.set_ylabel('Time (s)',fontsize=15)
      ax.tick_params(labelsize=15)
      plt.show()
    ctr += 1

printprogress("nshots:",nsy*nsx,nsy*nsx)
# Close all of the opened files
for kvp in dmap.items():
  kvp[1].close()

