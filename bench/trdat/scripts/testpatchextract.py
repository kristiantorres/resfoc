import inpout.seppy as seppy
import deeplearn.utils as dlut
from deeplearn.python_patch_extractor.PatchExtractor import PatchExtractor
import matplotlib.pyplot as plt

# Read in the seismic image
sep = seppy.sep([])
iaxes,img = sep.read_file(None,ifname='./dat/vels/velfltimg0000.H')
img = img.reshape(iaxes.n,order='F')

laxes,lbl= sep.read_file(None,ifname='./dat/vels/velfltlbl0000.H')
lbl = lbl .reshape(laxes.n,order='F')

imgwind = img[:,:,0].T
lblwind = lbl[:,:,0].T

# Interpolate the output image and the label
imgwind = (dlut.resample(imgwind,[1024,512],kind='linear')).T
lblwind = (dlut.thresh(dlut.resample(lblwind,[1024,512],kind='linear'),0)).T

patch_shape = (128, 128)
stride = (64, 64)
pe = PatchExtractor(patch_shape,stride=stride)

iptch = pe.extract(imgwind)
lptch = pe.extract(lblwind)

pz = iptch.shape[0]
px = iptch.shape[1]
print(pz,px)

niptch = iptch.reshape([pz*px,128,128])
nlptch = lptch.reshape([pz*px,128,128])

print(iptch.shape)
print(niptch.shape)

plt.figure(1)
#plt.imshow(iptch[4,3],cmap='gray')
#dlut.plotseglabel(iptch[4,3],lptch[4,3])
plt.imshow(niptch[30],cmap='gray')
dlut.plotseglabel(niptch[30],nlptch[30])
plt.show()

# Reconstruction
imrec = pe.reconstruct(iptch)
lbrec = pe.reconstruct(lptch)

plt.figure(3)
plt.imshow(imrec,cmap='gray')
dlut.plotseglabel(imrec,lbrec)
plt.show()

