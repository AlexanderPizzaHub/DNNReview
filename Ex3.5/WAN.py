from re import L
import torch
import numpy as np
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from torch.autograd import Variable
import pickle as pkl
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from utils import tools,model,pde,validation
import time


torch.set_default_dtype(torch.float64)

#name = 'uniform_lfbgs_simul_t1/'
dataname = '10000pts'
path = 'results/WAN/t6/'

bw = 1000000
val_interval = 50

Ksol = 3
Kadv = 1

max_outer_iter = 2800

solNet = model.NN()
advNet = model.adverse() 

solNet.apply(model.init_weights)
advNet.apply(model.init_weights)

if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(path+"sol_plots/"):
    os.mkdir(path+"sol_plots/")

if not os.path.exists(path+'adv_plots/'):
    os.mkdir(path+"adv_plots/")


with open("dataset/"+dataname,'rb') as pfile:
    d_c = pkl.load(pfile)
    b_c = pkl.load(pfile)
print(d_c.shape,b_c.shape)

intx1,intx2,intx3,intx4,intx5,intx6 = np.split(d_c,6,axis=1)
bdx1,bdx2,bdx3,bdx4,bdx5,bdx6 = np.split(b_c,6,axis=1)

tintx1,tintx2,tintx3,tintx4,tintx5,tintx6,tbdx1,tbdx2,tbdx3,tbdx4,tbdx5,tbdx6 = tools.from_numpy_to_tensor([intx1,intx2,intx3,intx4,intx5,intx6,bdx1,bdx2,bdx3,bdx4,bdx5,bdx6],[True,True,True,True,True,True,False,False,False,False,False,False],dtype=torch.float64)



#For simul, no cost evaluation, and we need data on whole domain.

#tdx,tdy,tbx,tby= tools.from_numpy_to_tensor([dx,dy,bx,by],[True,True,False,False])


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdrynp = pkl.load(pfile)

f,ygt,bdrydat = tools.from_numpy_to_tensor([f_np,y_gt,bdrynp],[False,False,False])

print(f)
f = f.reshape(-1,1)
bdrydat = bdrydat.reshape(-1,1)

print(f.shape,bdrydat.shape)


def L2InnerProd(u,v,x1,x2,x3,x4,x5,x6):
    u_values = u(x1,x2,x3,x4,x5,x6) 
    if type(v) is torch.Tensor:
        v_values = v 
    else:
        v_values = v(x1,x2,x3,x4,x5,x6) 
    return torch.mean(torch.mul(u_values,v_values))

def A(sol,adv,rhs,x1,x2,x3,x4,x5,x6):
    sol_out = sol(x1,x2,x3,x4,x5,x6)
    sol_x1 = torch.autograd.grad(sol_out.sum(),x1,create_graph=True)[0]
    sol_x2 = torch.autograd.grad(sol_out.sum(),x2,create_graph=True)[0]
    sol_x3 = torch.autograd.grad(sol_out.sum(),x3,create_graph=True)[0]
    sol_x4 = torch.autograd.grad(sol_out.sum(),x4,create_graph=True)[0]
    sol_x5 = torch.autograd.grad(sol_out.sum(),x5,create_graph=True)[0]
    sol_x6 = torch.autograd.grad(sol_out.sum(),x6,create_graph=True)[0]

    adv_out = adv(x1,x2,x3,x4,x5,x6)
    adv_x1 = torch.autograd.grad(adv_out.sum(),x1,create_graph=True)[0]
    adv_x2 = torch.autograd.grad(adv_out.sum(),x2,create_graph=True)[0]
    adv_x3 = torch.autograd.grad(adv_out.sum(),x3,create_graph=True)[0]
    adv_x4 = torch.autograd.grad(adv_out.sum(),x4,create_graph=True)[0]
    adv_x5 = torch.autograd.grad(adv_out.sum(),x5,create_graph=True)[0]
    adv_x6 = torch.autograd.grad(adv_out.sum(),x6,create_graph=True)[0]

    lhs_itg = torch.mean(sol_x1 * adv_x1 + sol_x2 * adv_x2 + sol_x3 * adv_x3 + sol_x4 * adv_x4 + sol_x5 * adv_x5 + sol_x6 * adv_x6)
    rhs_itg = L2InnerProd(adv,rhs,x1,x2,x3,x4,x5,x6)

    return lhs_itg - rhs_itg


def compute_lossmin():
    adv_norm_2 = L2InnerProd(advNet,advNet,tintx1,tintx2,tintx3,tintx4,tintx5,tintx6)
    A_value = A(solNet,advNet,f,tintx1,tintx2,tintx3,tintx4,tintx5,tintx6)

    #L_int = torch.square(A_value) / adv_norm_2
    L_int = torch.log(torch.square(A_value)) - torch.log(adv_norm_2)
    #Compute L_bdry
    bdry_out = solNet(tbdx1,tbdx2,tbdx3,tbdx4,tbdx5,tbdx6)

    L_bdry = torch.square(bdry_out - bdrydat).mean()

    #Compute the loss
    loss_min = L_int + bw * L_bdry

    return loss_min.detach().numpy(),L_int.detach().numpy(),L_bdry.detach().numpy()


#optimizer_sol = opt.LBFGS(solNet.parameters(),stephook=hook,line_search_fn="strong_wolfe",max_iter=200,max_eval=200,tolerance_grad=1e-15, tolerance_change=1e-15, history_size=100)
#optimizer_adv = opt.LBFGS(advNet.parameters(),line_search_fn="strong_wolfe",max_iter=100,max_eval=100,tolerance_grad=1e-15, tolerance_change=1e-15, history_size=100)
#optimizer_sol = opt.Adam(solNet.parameters(),lr=1e-4)
#optimizer_adv = opt.Adam(advNet.parameters(),lr=1e-3)
optimizer_sol = opt.Adagrad(solNet.parameters(),lr=0.015)
optimizer_adv = opt.Adagrad(advNet.parameters(),lr=0.04)

def closure_sol():
    optimizer_sol.zero_grad()
    optimizer_adv.zero_grad()

    #Compute L_int 
    adv_norm_2 = L2InnerProd(advNet,advNet,tintx1,tintx2,tintx3,tintx4,tintx5,tintx6)
    A_value = A(solNet,advNet,f,tintx1,tintx2,tintx3,tintx4,tintx5,tintx6)
    L_int = torch.square(A_value) / adv_norm_2
    #L_int = torch.log(torch.square(A_value)) - torch.log(adv_norm_2)

    #Compute L_bdry
    bdry_out = solNet(tbdx1,tbdx2,tbdx3,tbdx4,tbdx5,tbdx6)

    L_bdry = torch.square(bdry_out - bdrydat).mean()

    #Compute the loss
    loss_min = L_int + bw * L_bdry
    #print(L_int,L_bdry)
    #Backward 
    #print(loss_min)
    loss_min.backward()

    return loss_min.detach().numpy()

def closure_adv():
    optimizer_sol.zero_grad()
    optimizer_adv.zero_grad()

    #Compute L_int 
    adv_norm_2 = L2InnerProd(advNet,advNet,tintx1,tintx2,tintx3,tintx4,tintx5,tintx6)
    A_value = A(solNet,advNet,f,tintx1,tintx2,tintx3,tintx4,tintx5,tintx6)

    L_int = torch.square(A_value) / adv_norm_2
    #L_int = torch.log(torch.square(A_value)) - torch.log(adv_norm_2)
    
    #Compute the loss
    loss_max = - L_int

    #Backward 
    loss_max.backward()

    return loss_max.detach().numpy()


vlist = [] 
losslist = []
if not os.path.exists(path+"y_plot/"):
    os.makedirs(path+"y_plot/")
for iter in range(max_outer_iter):
    for isol in range(Ksol):
        optimizer_sol.step(closure_sol)
    for iadv in range(Kadv):
        optimizer_adv.step(closure_adv)

    if iter % 10 == 0:
        vy,_,_ = validation.validate(solNet)
        loss = float(closure_sol())
        torch.save(solNet,path+"solNet.pt")
        validation.plot_2D(solNet,path+"y_plot/"+'y')
        print("At iteration:{}, val:{}".format(iter,float(vy)))

        vlist.append(float(vy))
        losslist.append(loss)

with open(path+"vlist",'wb') as pfile:
    pkl.dump(vlist,pfile)
with open(path+"losslist",'wb') as pfile:
    pkl.dump(losslist,pfile)