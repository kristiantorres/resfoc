import getpass
import subprocess
import time
import pyslurm

ntry = 50
begtimef = time.time()
for itry in range(ntry):
  cmd = 'squeue -a -o "%.18i %.9P %.17j %.10u %.2t %.10M %.6D %R" > squeue.out'
  sp = subprocess.check_call(cmd,shell=True)
  with open('squeue.out','r') as f:
    info = f.readlines()
  if(len(info) == 1):
    raise Exception("Must start workers before checking their status")
  # Remove the header
  del info[0]
print(time.time() - begtimef)

begtime = time.time()
for itry in range(ntry):
  cmd = 'squeue -a -o "%.18i %.9P %.17j %.10u %.2t %.10M %.6D %R"'
  sp = subprocess.Popen(['squeue','-a','-o','%.18i %.9P %.17j %.10u %.2t %.10M %.6D %R'],stdout=subprocess.PIPE)
  out,err = sp.communicate()
  info = out.decode("utf-8").split("\n")
  if(len(info) == 1):
    raise Exception("Must start workers before checking their status")
  # Remove the header
  del info[0]
print(time.time() - begtime)

begtimes = time.time()
for itry in range(ntry):
  info = pyslurm.job().get()
print(time.time() - begtimes)


