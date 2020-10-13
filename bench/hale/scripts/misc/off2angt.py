import inpout.seppy as seppy
import numpy as np

sep = seppy.sep()

iaxes,img = sep.read_file("haleres2.H")
[dz,dx,dhx,dro] = iaxes.d; [oz,ox,ohx,oro] = iaxes.o

img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T)

imgn = img[np.newaxis]

imgout = np.transpose(imgn,(1,3,0,2,4)) #[nhy,nro,nhx,nx,nz] -> [nro,nx,nhy,nhx,nz]

fxw = 100; nxw = 150
imgoutw = imgout[:,fxw:fxw+nxw,:,:,:]

sep.write_file("halres2t.H",imgoutw.T,ds=[dz,dhx,1.0,dx,dro],os=[oz,ohx,0.0,ox+fxw*dx,oro])


