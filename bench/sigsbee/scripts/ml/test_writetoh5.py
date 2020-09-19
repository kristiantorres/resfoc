import numpy as np
from deeplearn.dataloader import WriteToH5

# Create a simple dataset

ntot = 30 ; nbt = 123
xot = np.random.rand(nbt,64,64).astype('float32')
yot = np.random.rand(nbt,64,64).astype('float32')

wh5 = WriteToH5("/scr2/joseph29/h5test.h5",dsize=10)

for iex in range(ntot):
  wh5.write_examples(xot,yot)

