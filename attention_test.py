import torch
import numpy as np


# x = torch.rand((2, 50000, 2000)).to('cuda').float()
# y = torch.rand((2, 20000, 2000)).to('cuda').float()

# W = torch.bmm(x, y.transpose(2, 1))
# yy = torch.bmm(W, y)


# x = torch.tensor([[[[1, 2], [2, 3]],[[4, 5], [5, 6]]], [[[1, 2], [2, 3]],[[4, 5], [5, 6]]]])
# x = torch.rand((2, 256, 120, 160)).to('cuda').float()
# xr = x.view((x.shape[0], x.shape[1], -1))

x = torch.rand((8, 16, 3)).to('cuda').float()
y = torch.rand((8, 3, 12)).to('cuda').float()
z = torch.rand((8, 12, 4)).to('cuda').float()
w = torch.bmm(x, y)

def batch_matMul(x, y, z):
    w = torch.bmm(x, y)
    zw1 = torch.bmm(w, z)
    return zw1

def einsum_batch_matMul(x, y, z):
    zw2 = torch.einsum('bij,bjk,bkl->bil', [x, y, z])
    return zw2


# xm = torch.max(x, 1, keepdim=True)[0] 
# xr = xm.view(-1, 18)

# x = torch.tensor([[10, 0], [100, 0]]).float()



print('test')