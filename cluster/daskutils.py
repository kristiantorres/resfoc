"""
Utility functions to help with cluster
computing with Dask
@author: Joseph Jennings
@version: 2020.08.04
"""
import os,signal
import subprocess
import socket

def shutdown_sshcluster(hosts):
  """
  Kills all hanging processes left by a dask
  SSH cluster

  Parameters:
    hosts - a list of hostnames (the same that was passed to
            SSHCluster)
  """
  # Get the local hostname
  local = gethostname()
  # Command to scheduler processes
  for hname in hosts:
    if(hname == local or hname == 'localhost'):
      check_kill_process("distributed.cli.dask_")
    else:
      remkill = """ ssh -n -f %s "sh -c \\"pkill -f \\"distributed.cli.dask_\\"\\"" """%(hname)
      subprocess.check_call(remkill,shell=True)

def clean_slurm(path="."):
  """
  Cleans all of the slurm files generated from cluster submissions
  """
  subprocess.check_call('rm -f %s/slurm*.out'%(path),shell=True)

def check_kill_process(pstring):
  """ Kills a process given a string """
  for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"): 
    fields = line.split() 
    pid = fields[0] 
    os.kill(int(pid), signal.SIGKILL)

def gethostname(alias=True):
  """ 
  An extension of socket.gethostname() where the hostname alias
  is returned if requested
  """
  hname = socket.gethostname()
  if(alias):
    with open('/etc/hosts','r') as f:
      for line in f.readlines():
        cols = line.lower().split()
        if(len(cols) > 1): 
          if(hname == cols[1] and len(cols) > 2): 
            hname = cols[2]

  return hname

