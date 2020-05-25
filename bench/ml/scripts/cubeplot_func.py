import numpy as np
import h5py
from utils.plot import plot_cubeiso

#hf = h5py.File('/net/fantastic/scr2/joseph29/angfaultfocptch.h5')
hf = h5py.File('/net/fantastic/scr2/joseph29/angfaultdefptch.h5')

keys = list(hf.keys())

iptch = np.random.randint(105)
iex = np.random.randint(1000)
print(iex,iptch)
onex = hf[keys[iex]][iptch,:,:,:,0]

hf.close()

plot_cubeiso(onex,os=[-70.0,0.0,0.0],ds=[2.22,0.01,0.01],stack=True,show=True,hbox=8,wbox=4,elev=15,
             x1label='\nX (km)',x2label='\nAngle'+r'($\degree$)',x3label='\nDepth (km)')
