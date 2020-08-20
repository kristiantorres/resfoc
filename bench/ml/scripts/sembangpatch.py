import numpy as np
import h5py
from deeplearn.focuslabels import semblance_power
from scaas.trismooth import smooth
from genutils.plot import plot_cubeiso
import matplotlib.pyplot as plt

hff = h5py.File('/net/fantastic/scr2/joseph29/angfaultfocptch.h5')
hfd = h5py.File('/net/fantastic/scr2/joseph29/angfaultdefptch.h5')

keys = list(hff.keys())

iptch = np.random.randint(105)
iex = np.random.randint(1000)
#iptch = 1; iex = 266
print(iex,iptch)
foc = hff[keys[iex]][iptch,:,:,:,0]
dfc  = hfd[keys[iex]][iptch,:,:,:,0]

# Semblance calculation
fsemb = semblance_power(foc)
dsemb = semblance_power(dfc)

print(dsemb/fsemb)

os = [-70.0,0.0,0.0]; ds = [2.22,0.01,0.01]
plot_cubeiso(foc,os=os,ds=ds,stack=True,show=False,hbox=8,wbox=8,elev=15,
             x1label='\nX (km)',x2label='\nAngle '+r'($\degree$)',x3label='\nDepth (km)',verb=False)

plot_cubeiso(dfc,os=os,ds=ds,stack=True,show=True,hbox=8,wbox=8,elev=15,
             x1label='\nX (km)',x2label='\nAngle '+r'($\degree$)',x3label='\nDepth (km)',verb=False)

hff.close(); hfd.close()

