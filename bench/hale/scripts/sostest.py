import os,sys
from java.nio import *
from java.awt import *
from java.io import *
from java.lang import *
from java.util import *
from javax.swing import *

from edu.mines.jtk.awt import *
from edu.mines.jtk.dsp import *
from edu.mines.jtk.io import *
from edu.mines.jtk.mosaic import *
from edu.mines.jtk.sgl import *
from edu.mines.jtk.util import *
from edu.mines.jtk.util.ArrayMath import *

backgroundColor = Color(0xfd,0xfe,0xff) # easy to make transparent

def main(args):
  nx,nz = 560,900
  ais = ArrayInputStream("rfi.dat")
  x   = zerofloat(nz,nx)
  #xsm = zerofloat(nz,nx)
  ais.readFloats(x)
  ais.close()
  # Create the samplings
  sx = Sampling(nx,0.01675,7.37)
  sz = Sampling(nz,0.005,0.01675)
  x = mul(1e-8,x)
  # Compute the structure tensors
  lof = LocalOrientFilter(4.0)
  s = lof.applyForTensors(x)
  d00 = EigenTensors2(s); d00.invertStructure(0.0,0.0)
  d01 = EigenTensors2(s); d01.invertStructure(0.0,1.0)
  d02 = EigenTensors2(s); d02.invertStructure(0.0,2.0)
  d04 = EigenTensors2(s); d04.invertStructure(0.0,4.0)
  d11 = EigenTensors2(s); d11.invertStructure(1.0,1.0)
  d12 = EigenTensors2(s); d12.invertStructure(1.0,2.0)
  d14 = EigenTensors2(s); d14.invertStructure(1.0,4.0)
  xsm = copy(x)
  hw2 = 10
  c = hw2*(hw2+1)/6.0
  #smooth = LocalSmoothingFilter()
  #smooth.apply(d04,c,x,xsm)
  #smooth.applySmoothS(xsm,xsm)
  #hw = 20
  #lsf = LocalSemblanceFilter(hw,hw)
  #smb = lsf.semblance(LocalSemblanceFilter.Direction2.V,s,x)
  #smb = mul(smb,smb)
  #smb = mul(smb,smb)
  ##plotImg(smb,sx,sz)
  #s.setEigenvalues(0.0,1.0)
  #smooth = LocalSmoothingFilter()
  ##hw2 = 10
  #c = hw2*(hw2+1)/6.0
  #smooth.apply(s,c,smb,x,xsm)
  plotImgTensors("D04",x,sz,sx,d=d04)
  #xsm = mul(1e8,xsm)
  ## Write semblance
  #aos = ArrayOutputStream("smb.dat")
  #aos.writeFloats(smb)
  #aos.close()
  ## Write smoothed image
  #aos = ArrayOutputStream("out.dat")
  #aos.writeFloats(xsm)
  #aos.close()

def plotImg(img,sx,sz):
  pp = PlotPanel(PlotPanel.Orientation.X1DOWN_X2RIGHT)
  pp.setBackground(backgroundColor)
  pp.setHLabel('X (km)')
  pp.setVLabel('Z (km)')
  pv = pp.addPixels(sz,sx,img)
  pv.setColorModel(ColorMap.GRAY)
  pv.setInterpolation(PixelsView.Interpolation.LINEAR)
  pv.setPercentiles(1,99)
  pf = PlotFrame(pp)
  pf.setSize(1000,700)
  pf.setVisible(True)

def plotImgTensors(title,img,s1,s2,d=None):
  sp = SimplePlot(SimplePlot.Origin.UPPER_LEFT)
  sp.setBackground(backgroundColor)
  sp.setHLabel('X (km)')
  sp.setVLabel('Z (km)')
  sp.setTitle(title)
  pv = sp.addPixels(s1,s2,img)
  pv.setColorModel(ColorMap.GRAY)
  pv.setInterpolation(PixelsView.Interpolation.LINEAR)
  pv.setPercentiles(1,99)
  if(d):
    tv = TensorsView(s1,s2,d)
    tv.setOrientation(TensorsView.Orientation.X1DOWN_X2RIGHT)
    tv.setLineColor(Color.YELLOW)
    tv.setLineWidth(3)
    tv.setEllipsesDisplayed(40)
    tv.setScale(1.0)
    tile = sp.plotPanel.getTile(0,0)
    tile.addTiledView(tv)
  pv.setVisible(True)

class RunMain(Runnable):
  def run(self):
    main(sys.argv)
SwingUtilities.invokeLater(RunMain())


