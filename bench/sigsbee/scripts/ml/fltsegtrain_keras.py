import os
import inpout.seppy as seppy
import numpy as np
from deeplearn.kerasnets import unetxwu
from deeplearn.dataloader import load_alldata
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from deeplearn.utils import plotseglabel
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

allx,ally = load_alldata("/net/thing/scr2/joseph29/sigsbee_fltseg.h5",None,1)
xshape = allx.shape[1:]
yshape = ally.shape[1:]

ntr = allx.shape[0]

#for itr in range(20):
#  iex = np.random.randint(ntr)
#  plt.figure()
#  plt.imshow(allx[iex,:,:,0],cmap='gray',interpolation='bilinear',aspect=0.5)
#  plotseglabel(allx[iex,:,:,0],ally[iex,:,:,0],aratio=0.5,interpolation='bilinear',show=True)

model = unetxwu(input_size=(64,64,1))

# Set GPUs
tf.compat.v1.GPUOptions(allow_growth=True)

# Create callbacks
checkpointer = ModelCheckpoint(filepath="/scr1/joseph29/sigsbee_fltseg-chkpt.h5", verbose=1, save_best_only=True)

# Train the model
nepochs = 10; bsize = 20
history = model.fit(allx,ally,epochs=nepochs,batch_size=bsize,verbose=1,shuffle=True,
                   validation_split=0.2,callbacks=[checkpointer])

# Write the model
model.save_weights("./dat/fltsegwgts.h5")

# Save the loss history
sep = seppy.sep()
lossvepch = np.asarray(history.history['loss'])
sep.write_file("fltsegloss.H",lossvepch)
vlssvepch = np.asarray(history.history['val_loss'])
sep.write_file("fltsegvlss.H",vlssvepch)

# Save the accuracy history
accvepch = np.asarray(history.history['acc'])
sep.write_file("fltsegacc.H",accvepch)
vacvepch = np.asarray(history.history['val_acc'])
sep.write_file("fltsegvac.H",vacvepch)

# Save the model architecture
with open("./dat/fltsegarch.json",'w') as f:
  f.write(model.to_json())

