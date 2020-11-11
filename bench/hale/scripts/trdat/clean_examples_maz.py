import inpout.seppy as seppy
import numpy as np
from genutils.ptyprint import progressbar

sep = seppy.sep()

taxes = sep.read_header("hale_foctrimgsmaz.H")
nz,na,naz,nx,nm = taxes.n
dz,da,daz,dx,dm = taxes.d
oz,oa,oaz,ox,om = taxes.o

# Examples to skip
skips = list(range(61,78))

for iex in progressbar(range(nm),"iex"):
  if(iex not in skips):
    # Read in one example
    faxes,foc = sep.read_wind("hale_foctrimgsmaz.H",fw=iex,nw=1)
    foc = foc.reshape(faxes.n,order='F')
    daxes,dfc = sep.read_wind("hale_deftrimgsmaz.H",fw=iex,nw=1)
    dfc = dfc.reshape(daxes.n,order='F')
    raxes,res = sep.read_wind("hale_restrimgsmaz.H",fw=iex,nw=1)
    res = res.reshape(raxes.n,order='F')
    saxes,sca = sep.read_wind("hale_randrhosmaz.H",fw=iex,nw=1)
    if(iex == 0):
      sep.write_file("hale_foctrimgsmazcln.H",foc,os=[oz,oa,0.0,ox],ds=[dz,da,1.0,dx])
      sep.write_file("hale_deftrimgsmazcln.H",dfc,os=[oz,oa,0.0,ox],ds=[dz,da,1.0,dx])
      sep.write_file("hale_restrimgsmazcln.H",res,os=[oz,oa,0.0,ox],ds=[dz,da,1.0,dx])
      sep.write_file("hale_randrhosmazcln.H",sca)
    elif(iex == 1):
      sep.append_file("hale_foctrimgsmazcln.H",foc,newaxis=True)
      sep.append_file("hale_deftrimgsmazcln.H",dfc,newaxis=True)
      sep.append_file("hale_restrimgsmazcln.H",res,newaxis=True)
      sep.append_file("hale_randrhosmazcln.H",sca)
    else:
      sep.append_file("hale_foctrimgsmazcln.H",foc)
      sep.append_file("hale_deftrimgsmazcln.H",dfc)
      sep.append_file("hale_restrimgsmazcln.H",res)
      sep.append_file("hale_randrhosmazcln.H",sca)

