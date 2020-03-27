import inpout.seppy as seppy
from utils.movie import makemovie_mpl

sep = seppy.sep([])

raxes,resmig = sep.read_file(None,ifname='trmigwrng.H')

nz  = raxes.n[0]; oz  = raxes.o[0]; dz  = raxes.d[0]
nx  = raxes.n[1]; ox  = raxes.o[1]; dx  = raxes.d[1]
nro = raxes.n[2]; oro = raxes.o[2]; dro = raxes.d[2]

resmig = resmig.reshape(raxes.n,order='F')

makemovie_mpl(resmig,'./fig/trmigwrng',xmin=0.0,xmax=(nx-1)*dx/1000.0,zmin=0.0,zmax=(nz-1)*dz/1000.0,
              xlabel='X (km)',ylabel='Z (km)',vmin=-4e-5,vmax=4e-5,qc=False,
              ttlstring=r'$\rho$=%.2f',ottl=oro,dttl=dro,pttag=True)

