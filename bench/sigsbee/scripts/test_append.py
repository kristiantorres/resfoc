import inpout.seppy as seppy
import numpy as np

sep = seppy.sep()

#a = np.zeros([100,50],dtype='float32')
#a[50] = 1.0
#b = np.zeros([100,50,2],dtype='float32')
#b[25,:,0] = 1.0
#b[75,:,1] = 1.0
#
#sep.write_file("a.H",a,os=[0.0,0.1],ds=[0.004,0.05])
#
#sep.append_file("a.H",b)

a = np.random.rand(100,50,10).astype('float32')
b = np.random.rand(100,50,2).astype('float32')

sep.write_file("a.H",a)
sep.append_file("a.H",b)

