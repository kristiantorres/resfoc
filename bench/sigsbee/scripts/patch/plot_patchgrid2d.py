import inpout.seppy as seppy
import numpy as np
from deeplearn.utils import plot_patchgrid2d, patch_window2d
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
import matplotlib.pyplot as plt

sep = seppy.sep()

iaxes,img = sep.read_file("resmskoverwro1.H")
oz,oa,ox = iaxes.o; dz,da,dx = iaxes.d
img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)
stk = np.sum(img,axis=1)

# Define window
bxw = 20;  exw = 480
bzw = 240; ezw = 1150

stkw = stk[bxw:exw,bzw:ezw].T

print(stkw.shape)

pe = PatchExtractor((64,64),stride=(32,32))
ptches = pe.extract(stkw)
print(ptches.shape)

stkb = pe.reconstruct(ptches)
print(stkb.shape)

imgw = patch_window2d(stkw,64,64)
print(imgw.shape)

plot_patchgrid2d(imgw,64,64,dz=dz,dx=dx,oz=0,ox=0,pclip=0.4)

