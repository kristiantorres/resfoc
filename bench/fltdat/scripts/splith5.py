from deeplearn.dataloader import splith5
import h5py

fin = '/scr2/joseph29/ptscatmigtest_copy.h5'
f1 = '/scr2/joseph29/testval1.h5'
f2 = '/scr2/joseph29/testval2.h5'

splith5(fin,f1,f2,rand=False)

hf1 = h5py.File(f1,'r')
print(len(list(hf1.keys()))/2)

hf2 = h5py.File(f2,'r')
print(len(list(hf2.keys()))/2)

