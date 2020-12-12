import inpout.seppy as seppy
import numpy as np
import pickle
import zlib, lz4.frame, zfpy

sep = seppy.sep()

# Read in the extended image
iaxes,img = sep.read_file("f3imgextcritical.H")
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32')

print("Compressing array")
zimg = zfpy.compress_numpy(img)

obj = {}
obj['result'] = zimg

print('Pickling')
# Pickle compressed and uncompressed
p = pickle.dumps(obj,protocol=-1)

# Compressing
print("Compressing")
z = lz4.frame.compress(p,compression_level=-1)

# Write both  to file
print('Writing uncompressed to file')
with open("/scr2/joseph29/uncomp.p",'wb') as f:
  f.write(p)

print("Writing compressed to file")
with open("/scr2/joseph29/comp.p",'wb') as f:
  f.write(z)

