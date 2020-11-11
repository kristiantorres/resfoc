import inpout.seppy as seppy
import numpy as np

sep = seppy.sep()
faxes,foc= sep.read_file("hale_foctrimgsmaz.H")
foc   = np.ascontiguousarray(foc.reshape(faxes.n,order='F').T).astype('float32')
nz,na,ny,nx,nm = faxes.n

# Remove 61 - 78
focnew = np.zeros([nm-17,nx,ny,na,nz],dtype='float32')

k = 0
for im in range(nm):
  if(im >= 61 and im <= 77):
    continue
  else:
    focnew[k] = foc[im]
    k += 1

del foc
sep.write_file("hale_foctrimgsmazcln.H",focnew.T,ds=faxes.d,os=faxes.o)


