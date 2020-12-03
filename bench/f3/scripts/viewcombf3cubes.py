import inpout.seppy as seppy
import numpy as np
from oway.costaper import costaper
from scaas.gradtaper import build_taper
from resfoc.gain import tpow, agc
from genutils.movie import viewcube3d

#XXX: Remember -  the code crashed at f3img05.H
#     read that one in and then add to it
#     the last image computed from
#     the next round

# f3img10.H is the sum of 6,7,8,9,10


# Nice image for figures - 6077.15 475.55

sep = seppy.sep()

# Read in f3img05.H and f3img10.H
caxes,cub1 = sep.read_file("./f3tmps/f3img05.H")
dz,dx,dy = caxes.d; oz,ox,oy = caxes.o
cub1 = cub1.reshape(caxes.n,order='F')

caxes, cub2 = sep.read_file("./f3tmps/f3img12.H")
cub2 = cub2.reshape(caxes.n,order='F')
cub = cub1 + cub2
pcub = np.ascontiguousarray(cub.T).astype('float32')
# AGC
acub = agc(pcub)

# Read in f3 cube
faxes,f3 = sep.read_file("migwt.T")
f3 = f3.reshape(faxes.n,order='F').T # [ny,nx,nz] -> [nz,nx,ny]
print(f3.shape)
f3w = f3[:900,:500,25:125]

#viewcube3d(f3w,ds=[dz,dx,dy],os=[oz,ox,oy],interp='bilinear',width3=1.0,show=False)
viewcube3d(acub.T,ds=[dz,dx,dy],os=[oz,ox,oy],interp='bilinear',pclip=0.1,width3=1.0,show=True)

