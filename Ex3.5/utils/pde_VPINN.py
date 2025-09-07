import torch
import numpy as np
import torch.optim as opt
import pickle
import pandas as pd
from utils import model
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

def laplacian(x1,x2,net):
    x = torch.cat([x1,x2],dim=1)
    out = net(x)

    u_x1 = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx1 = torch.autograd.grad(u_x1.sum(),x1,create_graph=True)[0]

    u_x2 = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_xx2 = torch.autograd.grad(u_x2.sum(),x2,create_graph=True)[0]

    return u_xx1+u_xx2

def pde_lhs(net,qp,w,cm):
    out = net(qp)
    grad_u = torch.autograd.grad(out.sum(),qp,create_graph=True)[0] #shape: (nrElem,6,2) Last dimension refers to u_x and u_y. Correct after check.

    #\[sum_i (w_i u_x(x_i))] * b_j + \[sum_i (w_i u_y(x_i))] * c_j for j = 1,2,3. Three vertices.

    lhs = torch.mul(torch.sum(w * grad_u[:,:,0].unsqueeze(-1),dim=1), cm[:,:,1]) +torch.mul(torch.sum(w * grad_u[:,:,1].unsqueeze(-1),dim=1), cm[:,:,2]) #shape:(nrElem,3)
    #print(lhs.shape)
    return lhs.unsqueeze(-1)

def pde_rhs(f_val,qp,w,cm):
    #f_val = f(qp).unsqueeze(-1)
    rhs = torch.sum(f_val * w,dim=1)*cm[:,:,0] + torch.sum(f_val*w*qp[:,:,0].unsqueeze(-1),dim=1)*cm[:,:,1]+torch.sum(f_val*w*qp[:,:,1].unsqueeze(-1),dim=1)*cm[:,:,2]
    return rhs.unsqueeze(-1)


def pde_bdry(net,bdry_col):
    out = net(bdry_col)
    return out


