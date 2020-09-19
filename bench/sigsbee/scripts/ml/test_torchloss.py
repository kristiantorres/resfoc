import torch
from deeplearn.torchlosses import bal_ce

tgt = torch.tensor([ 1, 1, 0, 1, 0,0 ],dtype=torch.float)
lgt = torch.tensor([10,30, 7, 11.2, 5,15],dtype=torch.float)

lossfn = bal_ce()

a = lossfn(lgt,tgt)

n = 32
target = torch.randint(high=2,size=[1,n]).float()
print(target.size())
for i in range(1000):
  inp = torch.rand([1,n],requires_grad=True)
  print(inp.size())
  loss = lossfn(inp,target)
  loss.backward()
  if(torch.isnan(loss)):
    print('Loss NaN')
  if(torch.isnan(inp.grad).any()):
    print('NaN')
