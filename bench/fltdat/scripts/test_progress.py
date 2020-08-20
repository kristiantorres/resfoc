from genutils.ptyprint import update_progress, progressbar
import time

# First one
print("progress : 0->1")
for i in range(101):
  time.sleep(0.1)
  update_progress(i/100.0)

print("")
print("Test completed")
time.sleep(2)

# Next one
for i in progressbar(range(15), "Computing: ", 40):
  time.sleep(0.1) # any calculation you need
