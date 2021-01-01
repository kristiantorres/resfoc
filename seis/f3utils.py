"""
Utility functions for manipulating/processing the F3
dataset

@author: Joseph Jennings
@version: 2020.12.21
"""
import numpy as np
from oway.mute import mute
import segyio
from genutils.ptyprint import progressbar
import subprocess
import matplotlib.pyplot as plt

def mute_f3shot(dat,isrcx,isrcy,nrec,strm,recx,recy,tp=0.5,vel=1450.0,dt=0.004,dx=0.025,
                hyper=True) -> np.ndarray:
  """
  Mutes a shot from the F3 dataset

  Parameters:
    dat   - an input shot gather from the F3 dataset [ntr,nt]
    isrcx - x source coordinate of the shot [float]
    isrcy - y source coordinate of the shot [float]
    strm  - index within the streamer
    recx  - x receiver coordinates for this shot [ntr]
    recy  - y receiver coordinates for this shot [ntr]
    vel   - water velocity [1450.0]
    tp    - length of taper [0.5s]
    dy    - minimum distance between streamers [20 m]
    dt    - temporal sampling interval [0.002]
    dx    - spacing between receivers [25 m]

  Returns a muted shot gather
  """
  mut = np.zeros(dat.shape,dtype='float32')
  v0 = vel*0.001
  # Find the beginning indices of the streamer
  idxs = list(np.where(strm[:nrec] == 1)[0])
  idxs.append(nrec)
  for istr in range(1,len(idxs)):
    irecx,irecy = recx[idxs[istr-1]],recy[idxs[istr-1]]
    dist = np.sqrt((isrcx-irecx)**2 + (isrcy-irecy)**2)
    t0 = dist/vel
    if(t0 > 0.15):
      t0 = dist/(vel)
      v0 = 1.5
    else:
      v0 = vel*0.001
    mut[idxs[istr-1]:idxs[istr]] = np.squeeze(mute(dat[idxs[istr-1]:idxs[istr]],dt=dt,dx=dx,v0=v0,t0=t0,tp=tp,
                                                   half=False,hyper=hyper))
  return mut

def select_f3shot(sx,sy,hdrkeys=None,hmap=None,allkeys=False,verb=True):
  """
  Selects an F3 shot and the header keys provided

  Parameters:
    sx      - source x coordinate
    sy      - source y coordinate
    hdrkeys - a list of header keys ['GroupX','GroupY','CDP_TRACE']
    hmap    - the hashmap for finding the files [None]
    allkeys - gives all the keys
    verb    - display a progressbar [True]

  Returns the traces for the desired shot as well as the
  desired headers
  """
  # Create the key for the hash map
  key = str(int(sy)) + ' ' + str(int(sx))
  if(hmap is None):
    # Read in hmap from file
    hmap = np.load('/data3/northsea_dutch_f3/segy/info/scoordhmap.npy',allow_pickle=True)[()]
  # Read in all source coordinates
  crds = np.load('/data3/northsea_dutch_f3/segy/info/scoords.npy',allow_pickle=True)[()]
  if(key not in hmap):
    #TODO: put in the option to find the nearest source
    #      will need to use the
    raise Exception("Provided source coordinate is not an actual coordinate")
  # Create the list of keys
  if(allkeys):
    hdrkeys = []
    for hkey in hdict: hdrkeys.append(hkey)
  if(hdrkeys is None):
    hdrkeys = []
    hdrkeys.append('GroupX');  hdrkeys.append('GroupY')
    hdrkeys.append('CDP_TRACE')
  # Get the file(s) that contain the data
  dmap = {}
  ohdict,srcdat = {},[]
  for ifile in hmap[key]:
    fname = '/data3/northsea_dutch_f3' + ifile[1:]
    if(fname not in dmap):
      dmap[fname] = segyio.open(fname,ignore_geometry=True)
    # First get the source coordinates for the file
    srcxf = np.asarray(dmap[fname].attributes(segyio.TraceField.SourceX),dtype='int32')
    srcyf = np.asarray(dmap[fname].attributes(segyio.TraceField.SourceY),dtype='int32')
    # Find the traces/indices associated with that source coordinate
    scoordsf = np.zeros([len(srcxf),2],dtype='int32')
    scoordsf[:,0] = srcyf; scoordsf[:,1] = srcxf
    idx1 = scoordsf == np.asarray([sy,sx],dtype='int32')
    s = np.sum(idx1,axis=1)
    nidx1 = s == 2
    # Header dictionaries
    for ikey in progressbar(hdrkeys,"keys:",verb=verb):
      if(ikey not in ohdict): ohdict[ikey] = []
      # Get the header values
      ohdict[ikey].append(np.asarray(dmap[fname].attributes(hdict[ikey]),dtype='int32')[nidx1])
    # Read in the data
    data = dmap[fname].trace.raw[:]
    srcdat.append(data[nidx1,:])
  # Concatenate the keys from different files
  for ikey in ohdict: ohdict[ikey] = np.concatenate(ohdict[ikey],axis=0)
  # Get the data for this shot
  srcdat = np.concatenate(srcdat,axis=0)

  # Return the data and the keys
  return ohdict,srcdat

def select_f3shotcont(fname,srcy,srcx,recy,recx,nrec):
  pass

def compute_batches(batchin,totnsht):
  """
  Computes the starting and stoping points for reading in
  batches from the F3 data file.

  Parameters:
    batchin - target batch size
    totnsht - total number of shots to read in

  Returns the batch size and the start and end of
  each batch
  """
  divs = np.asarray([i for i in range(1,totnsht) if(totnsht%i == 0)])
  bsize = divs[np.argmin(np.abs(divs - batchin))]
  nb = totnsht//bsize

  return bsize,nb

def compute_batches_var(batchin,totnsht):
  """
  Computes a variable batch size

  Parameters:
    batchin - the batch size (except at the end)
    totnsht - total number of shots to be read in

  Returns a list of batch sizes
  """
  igr = divmod(totnsht,batchin)
  if(igr[1] != 0):
    return [batchin]*igr[0] + [igr[1]]
  else:
    return [batchin]*igr[0]

def plot_acq(srcx,srcy,recx,recy,slc,ox,oy,
             dx=0.025,dy=0.025,srcs=True,recs=False,figname=None,**kwargs):
  """
  Plots the acqusition geometry on a depth/time slice

  Parameters:
    srcx    - source x coordinates
    srcy    - source y coordinates
    recx    - receiver x coordinatesq
    recy    - receiver y coordinates
    slc     - time or depth slice [ny,nx]
    ox      - slice x origin
    oy      - slice y origin
    dx      - slice x sampling [0.025]
    dy      - slice y sampling [0.025]
    recs    - plot only the receivers (toggles on/off the receivers)
    cmap    - 'grey' (colormap grey for image, jet for velocity)
    figname - output name for figure [None]
  """
  ny,nx = slc.shape
  cmap = kwargs.get('cmap','gray')
  fig = plt.figure(figsize=(14,7)); ax = fig.gca()
  ax.imshow(np.flipud(slc),cmap=cmap,extent=[ox,ox+nx*dx,oy,oy+ny*dy])
  if(srcs):
    ax.scatter(srcx,srcy,marker='*',color='tab:red')
  if(recs):
    ax.scatter(recx,recy,marker='v',color='tab:green')
  ax.set_xlabel('X (km)',fontsize=kwargs.get('fsize',15))
  ax.set_ylabel('Y (km)',fontsize=kwargs.get('fsize',15))
  ax.tick_params(labelsize=kwargs.get('fsize',15))
  if(figname is not None):
    plt.savefig(figname,dpi=150,transparent=True,bbox_inches='tight')
  if(kwargs.get('show',True)):
    plt.show()

def sum_extimgs(migdir,fout):
  """
  Sums partial extended images to form the full F3 image

  Parameters:
    migdir - directory containing migration images (string)
    fout   - output file that will contain the output image (string)

  Returns nothing
  """
  pyexec = "/sep/joseph29/anaconda3/envs/py37/bin/python"
  summer = "/homes/sep/joseph29/projects/resfoc/bench/f3/scripts/mig/MigSum.py"
  subprocess.Popen([pyexec,summer,"-migdir",migdir,"-fout",fout])

hdict={'TRACE_SEQUENCE_LINE': 1,
 'TRACE_SEQUENCE_FILE': 5,
 'FieldRecord': 9,
 'TraceNumber': 13,
 'EnergySourcePoint': 17,
 'CDP': 21,
 'CDP_TRACE': 25,
 'TraceIdentificationCode': 29,
 'NSummedTraces': 31,
 'NStackedTraces': 33,
 'DataUse': 35,
 'offset': 37,
 'ReceiverGroupElevation': 41,
 'SourceSurfaceElevation': 45,
 'SourceDepth': 49,
 'ReceiverDatumElevation': 53,
 'SourceDatumElevation': 57,
 'SourceWaterDepth': 61,
 'GroupWaterDepth': 65,
 'ElevationScalar': 69,
 'SourceGroupScalar': 71,
 'SourceX': 73,
 'SourceY': 77,
 'GroupX': 81,
 'GroupY': 85,
 'CoordinateUnits': 89,
 'WeatheringVelocity': 91,
 'SubWeatheringVelocity': 93,
 'SourceUpholeTime': 95,
 'GroupUpholeTime': 97,
 'SourceStaticCorrection': 99,
 'GroupStaticCorrection': 101,
 'TotalStaticApplied': 103,
 'LagTimeA': 105,
 'LagTimeB': 107,
 'DelayRecordingTime': 109,
 'MuteTimeStart': 111,
 'MuteTimeEND': 113,
 'TRACE_SAMPLE_COUNT': 115,
 'TRACE_SAMPLE_INTERVAL': 117,
 'GainType': 119,
 'InstrumentGainConstant': 121,
 'InstrumentInitialGain': 123,
 'Correlated': 125,
 'SweepFrequencyStart': 127,
 'SweepFrequencyEnd': 129,
 'SweepLength': 131,
 'SweepType': 133,
 'SweepTraceTaperLengthStart': 135,
 'SweepTraceTaperLengthEnd': 137,
 'TaperType': 139,
 'AliasFilterFrequency': 141,
 'AliasFilterSlope': 143,
 'NotchFilterFrequency': 145,
 'NotchFilterSlope': 147,
 'LowCutFrequency': 149,
 'HighCutFrequency': 151,
 'LowCutSlope': 153,
 'HighCutSlope': 155,
 'YearDataRecorded': 157,
 'DayOfYear': 159,
 'HourOfDay': 161,
 'MinuteOfHour': 163,
 'SecondOfMinute': 165,
 'TimeBaseCode': 167,
 'TraceWeightingFactor': 169,
 'GeophoneGroupNumberRoll1': 171,
 'GeophoneGroupNumberFirstTraceOrigField': 173,
 'GeophoneGroupNumberLastTraceOrigField': 175,
 'GapSize': 177,
 'OverTravel': 179,
 'CDP_X': 181,
 'CDP_Y': 185,
 'INLINE_3D': 189,
 'CROSSLINE_3D': 193,
 'ShotPoint': 197,
 'ShotPointScalar': 201,
 'TraceValueMeasurementUnit': 203,
 'TransductionConstantMantissa': 205,
 'TransductionConstantPower': 209,
 'TransductionUnit': 211,
 'TraceIdentifier': 213,
 'ScalarTraceHeader': 215,
 'SourceType': 217,
 'SourceEnergyDirectionMantissa': 219,
 'SourceEnergyDirectionExponent': 223,
 'SourceMeasurementMantissa': 225,
 'SourceMeasurementExponent': 229,
 'SourceMeasurementUnit': 231,
 'UnassignedInt1': 233,
 'UnassignedInt2': 237}

