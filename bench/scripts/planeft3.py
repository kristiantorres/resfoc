import inpout.seppy as seppy
import numpy as np
import cosft as cft
import matplotlib.pyplot as plt

# Create a cube
n1 = 100; n2 = 100; n3 = 100
cub = np.zeros([n1,n2,n3])
cub[:,49,:] = 1.0

cubft1   = cft.cosft(cub,axis1=1)
cubft2   = cft.cosft(cub,axis2=1)
cubft3   = cft.cosft(cub,axis3=1)
cubft12  = cft.cosft(cub,axis1=1,axis2=1)
cubft123 = cft.cosft(cub,axis1=1,axis2=1,axis3=1)
icubft123 = cft.icosft(cubft123,axis1=1,axis2=1,axis3=1)

sep = seppy.sep([])
axes = seppy.axes([n1,n2,n3],[0.0,0.0,0.0],[1.0,1.0,1.0])
sep.write_file(None,axes,cubft1,ofname='myft1.H')
sep.write_file(None,axes,cubft2,ofname='myft2.H')
sep.write_file(None,axes,cubft3,ofname='myft3.H')
sep.write_file(None,axes,cubft12,ofname='myft12.H')
sep.write_file(None,axes,cubft123,ofname='myft123.H')
sep.write_file(None,axes,icubft123,ofname='myift123.H')

