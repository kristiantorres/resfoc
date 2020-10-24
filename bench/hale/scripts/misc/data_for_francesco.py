import inpout.seppy as seppy
import numpy as np

sep = seppy.sep()

saxes,sht = sep.read_file("shots.H")
dt,do,dsx = saxes.d; ot,oo,osx = saxes.o
sht = sht.reshape(saxes.n,order='F').T

iaxes,img = sep.read_file("halerfi.H")
dz,dx = iaxes.d; oz,ox = iaxes.o
img = img.reshape(iaxes.n,order='F')
img = np.ascontiguousarray(img)

ddict = {}
ddict['dt'] = dt
ddict['ot'] = ot
ddict['doffset'] = do
ddict['ooffset'] = oo
ddict['dsx'] = dsx
ddict['osx'] = osx
ddict['data'] = sht 
np.save("data.npy",ddict)

idict = {}
idict['dz'] = dz
idict['dx'] = dx
idict['img'] = img[:600,:]
np.save('img.npy',idict)

