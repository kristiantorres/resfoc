import os, glob
import numpy as np

# SEGY directory
segdir = './segy/'

# Loop over all info files
files = sorted(glob.glob("./segy/info/*.txt"))

odict = {}; srcx = []; srcy = []
# Add each source coordinate to the hash map as the key
for ifile in files:
  # Read in in the file
  with open(ifile,'r') as f:
    scoords = f.readlines()
  # Change the file extension to SEGY
  sfile = segdir + os.path.splitext(os.path.basename(ifile))[0] + '.segy'
  for icrd in scoords:
    key = icrd.rstrip()[:14]
    srcx.append(int(key[:7])); srcy.append(int(key[8:]))
    if(key not in odict.keys()):
      odict[key] = [sfile]
    else:
      odict[key].append(sfile)

# Get unique source coordinates
srcx = np.asarray(srcx)
srcy = np.asarray(srcy)
srcs = np.zeros([len(srcx),2])
srcs[:,0] = srcx
srcs[:,1] = srcy
usrcs = np.unique(srcs,axis=0)

np.save('scoordhmap.npy',odict)
np.save('scoords.npy',usrcs)

