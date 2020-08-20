import numpy as np
import inpout.seppy as seppy
import matplotlib.pyplot as plt
from genutils.plot import plot_allanggats, plot_anggatrhos

sep = seppy.sep()

aaxes,angs = sep.read_file("rugimgsmallresmigang.H")

nz = aaxes.n[0]; na = aaxes.n[1]; nx = aaxes.n[2]; nro = aaxes.n[3]
dz = aaxes.d[0]; da = aaxes.d[1]; dx = aaxes.d[2]; dro = aaxes.d[3]
oz = aaxes.o[0]; oa = aaxes.o[1]; ox = aaxes.o[2]; oro = aaxes.o[3]

angs = angs.reshape(aaxes.n,order='F')

angst = angs.T
#plot_allanggats(angst[4],dz,dx,jx=4,interp='none')

plot_anggatrhos(angst,128,dz/1000.0,dx/1000.0,oro,dro)


