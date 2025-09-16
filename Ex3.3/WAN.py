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
dataname = '5000pts'
path = 'results/WAN/t7/'

bw = 1000000
val_interval = 50

Ksol = 3
Kadv = 1

max_outer_iter = 5000

solNet = model.NN_WAN()
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

dx,dy = np.split(d_c,2,axis=1)
bx,by = np.split(b_c,2,axis=1)


#For simul, no cost evaluation, and we need data on whole domain.

tdx,tdy,tbx,tby= tools.from_numpy_to_tensor([dx,dy,bx,by],[True,True,False,False])

print(tdx.shape,tbx.shape,"!!!")

with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdrynp = pkl.load(pfile)

f,ygt,bdrydat = tools.from_numpy_to_tensor([f_np,y_gt,bdrynp],[False,False,False])

print(f)
f = f.reshape(-1,1)
bdrydat = bdrydat.reshape(-1,1)

print(f.shape,bdrydat.shape)


def L2InnerProd(u,v,x,y):
    u_values = u(x,y) 
    if type(v) is torch.Tensor:
        v_values = v 
    else:
        v_values = v(x,y) 
    return torch.mean(torch.mul(u_values,v_values))

def A(sol,adv,rhs,x,y):
    sol_out = sol(x,y)
    sol_x = torch.autograd.grad(sol_out.sum(),x,create_graph=True)[0]
    sol_y = torch.autograd.grad(sol_out.sum(),y,create_graph=True)[0]

    adv_out = adv(x,y)
    adv_x = torch.autograd.grad(adv_out.sum(),x,create_graph=True)[0]
    adv_y = torch.autograd.grad(adv_out.sum(),y,create_graph=True)[0]

    lhs_itg = torch.mean(sol_x * adv_x + sol_y * adv_y)
    rhs_itg = L2InnerProd(adv,rhs,x,y)

    return lhs_itg - rhs_itg


def compute_lossmin():
    adv_norm_2 = L2InnerProd(advNet,advNet,tdx,tdy)
    A_value = A(solNet,advNet,f,tdx,tdy)

    #L_int = torch.square(A_value) / adv_norm_2
    L_int = torch.log(torch.square(A_value)) - torch.log(adv_norm_2)
    #Compute L_bdry
    bdry_out = solNet(tbx,tby)

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
    adv_norm_2 = L2InnerProd(advNet,advNet,tdx,tdy)
    A_value = A(solNet,advNet,f,tdx,tdy)
    L_int = torch.square(A_value) / adv_norm_2
    #L_int = torch.log(torch.square(A_value)) - torch.log(adv_norm_2)

    #Compute L_bdry
    bdry_out = solNet(tbx,tby)

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
    adv_norm_2 = L2InnerProd(advNet,advNet,tdx,tdy)
    A_value = A(solNet,advNet,f,tdx,tdy)

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

    if iter % 100 == 0:
        vy,_,_ = validation.validate(solNet)
        loss = float(closure_sol())
        torch.save(solNet,path+"solNet"+str(iter)+".pt")
        validation.plot_2D(solNet,path+"y_plot/"+'y')
        print("At iteration:{}, val:{}".format(iter,float(vy)))

        vlist.append(float(vy))
        losslist.append(loss)

        with open(path+"vlist",'wb') as pfile:
            pkl.dump(vlist,pfile)
        with open(path+"losslist",'wb') as pfile:
            pkl.dump(losslist,pfile)

with open(path+"vlist",'wb') as pfile:
    pkl.dump(vlist,pfile)
with open(path+"losslist",'wb') as pfile:
    pkl.dump(losslist,pfile)