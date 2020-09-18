import os
import inpout.seppy as seppy
import numpy as np
from deeplearn.dataloader import load_alldata
from deeplearn.kerasnets import vgg3_3d, vgg3_3d2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import random
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt
from genutils.plot import plot_cubeiso

sep = seppy.sep()

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# Load all data
focdat,foclbl = load_alldata("/net/thing/scr2/joseph29/sigsbee_fltfoc.h5",None,1,begex=0,endex=5000,)
resdat,reslbl = load_alldata("/net/thing/scr2/joseph29/sigsbee_fltres.h5",None,1,begex=0,endex=5000)
defdat,deflbl = load_alldata("/net/thing/scr2/joseph29/sigsbee_fltdeflbl.h5",None,1,begex=0,endex=5000)

print(focdat.shape,foclbl.shape)
print(defdat.shape,deflbl.shape)
print(resdat.shape,reslbl.shape)

# Concatenate focused and defocused images and labels
#allx = np.concatenate([focdat,resdat,defdat],axis=0)[:,32:,:,:,:]
allx = np.concatenate([focdat,resdat,defdat],axis=0)
ally = np.concatenate([foclbl[:,:,0],reslbl[:,:,0],deflbl],axis=0)

allx,ally = shuffle(allx,ally,random_state=1992)

print(allx.shape, ally.shape)

ntot = allx.shape[0]

# QC the images
os = [0.0,0.0,0.0]; ds = [1.875,0.00762,0.0457199]
for iex in range(10):
  idx = np.random.randint(ntot)
  plot_cubeiso(allx[idx,:,:,:,0],os=os,ds=ds,elev=15,verb=False,show=False,
      x1label='\nX (km)',x2label='\nAngle '+r'($\degree$)',x3label='Z (km)',stack=True)
  if(ally[idx] == 0):
    print("Defocused")
    #plt.savefig('./fig/defocused%d.png'%(idx),bbox_inches='tight',transparent=True,dpi=150)
  elif(ally[idx] == 1):
    print("Focused")
    #plt.savefig('./fig/focused%d.png'%(idx),bbox_inches='tight',transparent=True,dpi=150)
  else:
    print("Should not be here...")
  plt.show()
  plt.close()

#model = vgg3_3d2()
model = vgg3_3d()
print(model.summary())

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Create callbacks
checkpointer = ModelCheckpoint(filepath="/scr1/joseph29/sigsbee_focangchkpt.h5", verbose=1, save_best_only=True)

# Train the model
bsize   = 20
nepochs = 5
history = model.fit(allx,ally,epochs=nepochs,batch_size=bsize,verbose=1,shuffle=True,
                   validation_split=0.2,callbacks=[checkpointer])

# Write the model
model.save_weights("./dat/focangwgts.h5")

# Save the loss history
lossvepch = np.asarray(history.history['loss'])
sep.write_file("sigsbee_focanglss.H",lossvepch)
vlssvepch = np.asarray(history.history['val_loss'])
sep.write_file("sigsbee_focangvlss.H",vlssvepch)

# Save the accuracy history
accvepch = np.asarray(history.history['acc'])
sep.write_file("sigsbee_focangacc.H",accvepch)
vacvepch = np.asarray(history.history['val_acc'])
sep.write_file("sigsbee_focangvac.H",vacvepch)

# Save the model architecture
with open("./dat/focangarc.json",'w') as f:
  f.write(model.to_json())

