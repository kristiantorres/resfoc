"""
A wrapper for the structure-oriented smoothing Jython script

@author: Joseph Jennings
@version: 2020.11.02
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
from genutils.ptyprint import create_inttag
from client.sshworkers import create_host_list, launch_sshworkers, kill_sshworkers
from server.utils import startserver
from resfoc.soschunkr import soschunkr
from server.distribute import dstr_collect
from scaas.trismooth import smooth
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("-fin",help="input file",type=str)
ioArgs.add_argument("-fout",help="output file",type=str)
# Optional arguments
parser.add_argument("-labels",help="Input fault labels",type=str)
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep()

verb = sep.yn2zoo(args.verb)

faxes = sep.read_header(args.fin)
if(len(faxes.n) == 2):
  sosexe = "/homes/sep/joseph29/projects/resfoc/bench/hale/scripts/sos.py"
elif(len(faxes.n) == 3):
  sosexe = "/homes/sep/joseph29/projects/resfoc/bench/hale/scripts/sosang.py"
elif(len(faxes.n) == 4):
  sosexe = "/homes/sep/joseph29/projects/resfoc/bench/hale/scripts/sosang.py"
else:
  raise Exception("Can only handle up to 4-D inputs")

if(args.labels is not None):
  laxes,lbl = sep.read_file(args.labels)
  lbl = lbl.reshape(laxes.n,order='F').T
  zidx = lbl == 0
  oidx = lbl == 1
  lbl[zidx] = 1
  lbl[oidx] = -4.6
  smb = smooth(lbl.astype('float32'),rect1=20,rect2=20)
  osmb = "fltsmb.H"
  sep.write_file(osmb,smb.T,ofaxes=laxes)
  sos = "/sep/joseph29/jtk/bin/jy %s %s %s %s"%(sosexe,args.fin,osmb,args.fout)
  if(verb): print(sos)
  sp = subprocess.check_call(sos,shell=True)
else:
  if(len(faxes.n) == 2 or len(faxes.n) == 3):
    sos = "/sep/joseph29/jtk/bin/jy %s %s %s"%(sosexe,args.fin,args.fout)
    if(verb): print(sos)
    sp = subprocess.check_call(sos,shell=True)
  elif(len(faxes.n) == 4):
    # Read in the file
    iaxes,img = sep.read_file(args.fin)
    img = img.reshape(iaxes.n,order='F')
    oimg = np.zeros(img.T.shape)
    n4 = iaxes.n[-1]
    # Make directory
    tdir = os.getcwd() + '/sos4dtmp'
    if(not os.path.exists(tdir)):
      os.mkdir(tdir)
    # Split each of the 3D images
    imgs = []
    for k in range(n4):
      tag = create_inttag(k,1000)
      imgs.append("%s/img-%s.H"%(tdir,tag))
      if(not os.path.exists(imgs[k])):
        sep.write_file(imgs[k],img[:,:,:,k])
    # Start workers
    hosts = ['storm','torch','fantastic','jarvis']
    wph = len(hosts)*[3]
    hin = create_host_list(hosts,wph)
    cfile = "/homes/sep/joseph29/projects/resfoc/resfoc/sosworker.py"
    launch_sshworkers(cfile,hosts=hin,sleep=1,verb=1,clean=True)
    scnkr = soschunkr(n4,imgs,verb=False)
    gen = iter(scnkr)
    # Start the server
    context,socket = startserver()
    okeys = ['imgs']
    output = dstr_collect(okeys,n4,gen,socket,verb=True)
    oimgs = sorted(output['imgs'])
    # Clean up
    kill_sshworkers(cfile,hosts,verb=False)
    # Read in each into an array
    for k in range(len(oimgs)):
      saxes,simg = sep.read_file(oimgs[k][0])
      simg = simg.reshape(saxes.n,order='F').T
      oimg[k] = simg
    # Write out smoothed file
    sep.write_file(args.fout,oimg.T,ds=iaxes.d,os=iaxes.o)
    # Remove all temporary images
    #sp = subprocess.check_call("Rm %s/*.H"%(rdir),shell=True)
    #sp = subprocess.check_call("rmdir %s"%(tdir),shell=True)

