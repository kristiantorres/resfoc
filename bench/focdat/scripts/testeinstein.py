import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from resfoc.ssim import cwssim,ssimgauss
from pyssim.ssim.utils import to_grayscale
from skimage.measure import compare_ssim as ssim

size = (256,256)

im = Image.open('/home/joe/phd/projects/resfoc/bench/focdat/pyssim/test-images/test3-orig.jpg')
im = im.resize(size, Image.ANTIALIAS)
im_gray, alpha = to_grayscale(im)

# slightly rotated image
im_rot = Image.open('/home/joe/phd/projects/resfoc/bench/focdat/pyssim/test-images/test3-rot.jpg')
im_rot = im_rot.resize(size, Image.ANTIALIAS)
im_rot_gray, alpha = to_grayscale(im_rot)

# slightly modified lighting conditions
im_lig = Image.open('/home/joe/phd/projects/resfoc/bench/focdat/pyssim/test-images/test3-lig.jpg')
im_lig = im_lig.resize(size, Image.ANTIALIAS)
im_lig_gray, alpha = to_grayscale(im_lig)

# image cropped
im_cro = Image.open('/home/joe/phd/projects/resfoc/bench/focdat/pyssim/test-images/test3-cro.jpg')
im_cro = im_cro.resize(size, Image.ANTIALIAS)
im_cro_gray, alpha = to_grayscale(im_cro)

#print(cwssim(im,im_rot*0))
#print(cwssim(im,im_lig))
#print(cwssim(im,im_cro))

print(ssimgauss(im_gray,im_rot_gray))
print(ssimgauss(im_gray,im_lig_gray))
print(ssimgauss(im_gray,im_cro_gray))

print(" ")
print(ssim(im_gray,im_rot_gray,dynamic_range=np.max(im_rot_gray)-np.min(im_rot_gray)))
print(ssim(im_gray,im_lig_gray,dynamic_range=np.max(im_lig_gray)-np.min(im_lig_gray)))
print(ssim(im_gray,im_cro_gray,dynamic_range=np.max(im_cro_gray)-np.min(im_cro_gray)))

