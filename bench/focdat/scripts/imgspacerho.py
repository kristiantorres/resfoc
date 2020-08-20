import inpout.seppy as seppy
import numpy as np
from resfoc.estro import estro_tgt,refocusimg
from scaas.trismooth import smooth

sep = seppy.sep()

# Read in wrong zero subsurface offset
saxes,stk = sep.read_file("./fant/resfant3.H")
stk = stk.reshape(saxes.n,order='F')
stk = np.ascontiguousarray(stk.T).astype('float32')
stk = stk[:,16,:,:] # Take zero offset

nxw = 512; fx = 373
stkw = stk[:,fx:fx+nxw,:]

[nz,nx,nh,nro] = saxes.n; [oz,ox,oh,oro] = saxes.o; [dz,dx,dh,dro] = saxes.d

# Read in focused stack
faxes,foc = sep.read_file("./fant/imgafant3stk.H") # Angle stack
foc = foc.reshape(faxes.n,order='F')
foc = np.ascontiguousarray(foc.T).astype('float32')

rhoi = estro_tgt(stkw,foc,dro,oro,nzp=64,nxp=64,strdx=32,strdz=32)
rhoism = smooth(rhoi.astype('float32'),rect1=30,rect2=30)

sep.write_file('./fant/rhoifant3.H',rhoism.T,ds=[dz,dx])



