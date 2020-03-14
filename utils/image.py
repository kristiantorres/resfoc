"""
Utility functions for processing images

@author: Joseph Jennings
@version: 2020.03.14
"""
from PIL import Image

def remove_colorbar(ipath,cropsize,ftype='png',opath=None):
  """
  Crops the colorbar from an image while maintaining
  its size

  Parameters:
    impath   - path to the input image
    cropsize - how much to crop from the rightmost edge of the image
    opath    - path to the output image
  """
  im  = Image.open(ipath)
  isz = im.size
  im1 = im.crop((0,0,isz[0]-cropsize,isz[1]))
  # Padded image
  imp = Image.new('RGBA', isz, (255,0,0,0))
  imp.paste(im1,im1.getbbox())
  if(opath is None):
    base = ipath.split('.'+ftype)[0]
    opath = base + '-nocbar' + '.' + ftype
  imp.save(opath, ftype, quality=100)

