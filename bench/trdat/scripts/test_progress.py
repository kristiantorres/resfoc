from utils.ptyprint import update_progress, progressbar, printprogress
import time

## First one
#print("progress : 0->1")
#for i in range(101):
#  time.sleep(0.1)
#  update_progress(i/100.0)
#
#print("")
#print("Test completed")
#time.sleep(2)

# Next one
for i in progressbar(range(2), "Computing: ", 40):
  time.sleep(1) # any calculation you need

printprogress("nfaults",0,2,40)
time.sleep(1)
printprogress("nfaults",1,2,40)
time.sleep(1)
printprogress("nfaults",2,2,40)
