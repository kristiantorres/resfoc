import os
import h5py
import numpy as np
import tensorflow as tf
from deeplearn.dataloader import load_alldata, WriteToH5
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

#allx,ally = load_alldata("/scr1/joseph29/sigsbee_focdefres.h5",None,1)
allx,ally = load_alldata("/net/thing/scr2/joseph29/sigsbee_fltseg1.h5",None,1)
xshape = allx.shape[1:]
yshape = ally.shape[1:]

# Read in the network
with open('./dat/fltsegarch.json','r') as f:
  mdl = model_from_json(f.read())
mdl.load_weights('/scr1/joseph29/sigsbee_fltseg-chkpt.h5')

prd = mdl.predict(allx,verbose=True)

wh5 = WriteToH5('/net/thing/scr2/joseph29/sigsbee_fltseg1prds.h5',dsize=1)

wh5.write_examples(allx,prd)

