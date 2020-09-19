import torch
import numpy as np
from torch.nn import Conv2d
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt

nb   = 1
chin = 1; chot = 2

# Dimesions [batch,channels,height,width]
imgin = torch.from_numpy(np.zeros([nb,chin,100,100],dtype='float32'))
imgin[0,0,49,49] = 1.0

filt = np.zeros([chot,chin,3,3],dtype='float32')
filt[0,0,0,1] = 1
filt[0,0,1,0] = 1; filt[0,0,1,1] = -4; filt[0,0,1,2] = 1
filt[0,0,2,1] = 1
# Duplicate accross channel
#filt[1,0] = filt[0,0]

filtt = torch.from_numpy(filt)

convop = Conv2d(chin,chot,3,bias=False)

convop.weight = torch.nn.Parameter(filtt)

imgot = convop(imgin)

print(imgot.size())

imgotnp = imgot.detach().numpy()

imgotf = conv2d(imgin,filtt).numpy()

print(imgotf.shape)

plt.figure()
plt.imshow(imgotnp[0,1])
plt.figure()
plt.imshow(imgotf[0,1])
plt.show()

