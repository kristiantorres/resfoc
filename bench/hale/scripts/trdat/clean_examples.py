import inpout.seppy as seppy
import numpy as np

sep = seppy.sep()

taxes = sep.read_header("hale_foctrimgs.H")
nz,na,naz,nx,nm = taxes.n
dz,da,daz,dx,dm = taxes.d
oz,oa,oaz,ox,om = taxes.o

# Examples to skip
skips = [162]

for iex in range(progressbar(range(nm),"iex")):
  if(iex not in skips):
    # Read in one example
    faxes,foc = sep.read_wind("hale_foctrimgs",fw=iex,nw=1)
    foc = foc.reshape(faxes.n,order='F')
    daxes,dfc = sep.read_wind("hale_deftrimgs",fw=iex,nw=1)
    dfc = dfc.reshape(daxes.n,order='F')
    raxes,res = sep.read_wind("hale_restrimgs",fw=iex,nw=1)
    res = res.reshape(raxes.n,order='F')
    if(iex == 0):
      sep.write_file("hale_foctrimgscln.H",foc,os=[oz,oa,ox],ds=[dz,da,dx])
      sep.write_file("hale_deftrimgscln.H",dfc,os=[oz,oa,ox],ds=[dz,da,dx])
      sep.write_file("hale_restrimgscln.H",res,os=[oz,oa,ox],ds=[dz,da,dx])
    elif(iex == 1):
      sep.append_file("hale_foctrimgscln.H",foc,newaxis=True)
      sep.append_file("hale_deftrimgscln.H",dfc,newaxis=True)
      sep.append_file("hale_restrimgscln.H",res,newaxis=True)
    else:
      sep.append_file("hale_foctrimgscln.H",foc)
      sep.append_file("hale_deftrimgscln.H",dfc)
      sep.append_file("hale_restrimgscln.H",res)

