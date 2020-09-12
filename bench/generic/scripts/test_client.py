from client.slurmworkers import launch_slurmworkers, kill_slurmworkers, get_workers_times
import time

# Start workers
cfile = "/home/joseph29/projects/scaas/oway/hessnworker.py"
logpath = "./log"
wrkrs,status = launch_slurmworkers(cfile,wtime=30,nworkers=5,queue='sep',
                                   logpath=logpath,slpbtw=0.5,mode='adapt')

print("Status: ",*status)

time.sleep(20)

kill_slurmworkers(wrkrs)

