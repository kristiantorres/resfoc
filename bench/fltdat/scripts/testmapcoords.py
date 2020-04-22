import numpy
from scipy import ndimage 
from numpy.testing import assert_array_almost_equal

data = numpy.array([[4, 1, 3, 2],[7, 6, 8, 5],[3, 5, 3, 6]])
idx = numpy.indices(data.shape)
idx -= 1
print(data)
print(idx.shape)

print(idx)
expected = numpy.array([[0, 0, 0, 0],[0, 4, 1, 3],[0, 7, 6, 8]])

for order in range(0, 6):
  out = numpy.empty_like(expected)
  ndimage.map_coordinates(data, idx, order=order, output=out)
  assert_array_almost_equal(out, expected)
  
  out = numpy.empty_like(expected).astype(expected.dtype.newbyteorder())
  ndimage.map_coordinates(data, idx, order=order, output=out)
  assert_array_almost_equal(out, expected) 
