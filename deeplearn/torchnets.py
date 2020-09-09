import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
  """ Double convolution block """

  def __init__(self, chnsin, chnsot):
    super().__init()


class Down(nn.module):
  """ Downscaling with maxpool and then double conv """

  def __init__(self, chnsin, chnsot):
    super().__init__()
