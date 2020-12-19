import inpout.seppy as seppy
import time
import numpy as np
from genutils.ptyprint import progressbar

sep = seppy.sep()

beg = time.time()
for i in progressbar(range(50),"nfiles"):
  iaxes,img = sep.read_file("f3extimgs/f3extimg5m/f3extimg5m-00.H")
  img = img.reshape(iaxes.n,order='F')
  sep.write_file("iotest.H",img,os=iaxes.o,ds=iaxes.d)

print("Elapsed %f minutes"%((time.time()-beg)/60))


