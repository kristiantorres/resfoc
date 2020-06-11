import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt

sep = seppy.sep([])

laxes,loss = sep.read_file('./dat/nnhistory/fltclasswgt5-gpu10ep_loss.H')
vaxes,vlss = sep.read_file('./dat/nnhistory/fltclasswgt5-gpu10ep_vlss.H')
aaxes,accu = sep.read_file('./dat/nnhistory/fltclasswgt5-gpu10ep_accu.H')
caxes,vacu = sep.read_file('./dat/nnhistory/fltclasswgt5-gpu10ep_vacu.H')

fig = plt.figure(1,figsize=(7,7)); ax = fig.gca()
lin1 = plt.plot(loss/np.max(loss))
lin2 = plt.plot(vlss/np.max(vlss))
lin1.set_label('Loss',fontsize=18)
lin2.set_label('Validation',fontsize=18)
ax.set_xlabel('Epochs',fontsize=18)
ax.set_ylabel('Loss',fontsize=18)
ax.tick_params(labelsize=18)
ax.legend()
plt.savefig('./fig/ep10loss.png',bbox_inches='tight',dpi=150,transparent=True)

fig = plt.figure(2,figsize=(7,7)); ax = fig.gca()
lin1 = plt.plot(accu)
lin2 = plt.plot(vacu)
lin1.set_label('Accuracy',fontsize=18)
lin2.set_label('Validation',fontsize=18)
ax.set_xlabel('Epochs',fontsize=18)
ax.set_ylabel('Accuracy (%)',fontsize=18)
ax.tick_params(labelsize=18)
ax.legend()
plt.savefig('./fig/ep10accu.png',bbox_inches='tight',dpi=150,transparent=True)

