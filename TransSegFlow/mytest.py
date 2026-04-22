import torch
x=torch.tensor([[[1.,2.],[3.,4.],[4.,5.]],[[6.,7.],[8.,9.],[10.,11.]]])
v=torch.mean(x, dim=list(range(1, len(x.size()))))
choices = torch.rand(2, 5)
print(v)
print(choices)