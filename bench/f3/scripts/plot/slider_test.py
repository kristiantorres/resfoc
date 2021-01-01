# -*- coding: utf-8 -*-
"""
slider 3D numpy array

"""

import numpy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

data = numpy.random.rand(100,256,256) #3d-array with 100 frames 256x256

ax = plt.subplot(111)
plt.subplots_adjust(left=0.25, bottom=0.25)

frame = 0
l = plt.imshow(data[frame,:,:]) #shows 256x256 image, i.e. 0th frame

axcolor = 'lightgoldenrodyellow'
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
sframe = Slider(axframe, 'Frame', 0, 100, valinit=0)

def update(val):
  frame = int(numpy.around(sframe.val))
  plt.subplot(111)
  plt.subplots_adjust(left=0.25, bottom=0.25)
  plt.imshow(data[frame,:,:])

sframe.on_changed(update)

plt.show()

