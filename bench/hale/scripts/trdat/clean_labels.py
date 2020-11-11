import inpout.seppy as seppy
from deeplearn.dataloader import clean_examples

sep = seppy.sep()

skips = list(range(306,323))

files = ['hale_trlblsmaz.H']

clean_examples(files,skips,sep,verb=True)

