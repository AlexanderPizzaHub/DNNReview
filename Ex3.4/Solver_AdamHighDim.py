from time import time
from tracemalloc import start
import numpy as np
import torch
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from torch.autograd import Variable
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import model,pde,data,tools,g_tr
from time import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float32)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = 'cpu'



print(device)

y = model.NN()
y.apply(model.init_weights)

dataname = '10000pts'
name = 'results/'

bw = 60.0

if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"y_plot/"):
    os.makedirs(name+"y_plot/")


params = list(y.parameters())



with open("dataset/"+dataname,'rb') as pfile:
    int_col = pkl.load(pfile)
    bdry_col = pkl.load(pfile)

print(int_col.shape,bdry_col.shape)

dx1,dx2,dx3,dx4 = int_col.T
bx1,bx2,bx3,bx4 = bdry_col.T



tdc = tools.from_numpy_to_tensor_with_grad([dx1,dx2,dx3,dx4],device=device)
tbc = tools.from_numpy_to_tensor([bx1,bx2,bx3,bx4],device=device)


tdc_block = torch.stack(tdc,dim=1)
tbc_block = torch.stack(tbc,dim=1)


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdry_np = pkl.load(pfile)


f,bdrydat,ygt = tools.from_numpy_to_tensor([f_np,bdry_np,y_gt],device=device)



optimizer = opt.Adam(params,lr=1e-4)

mse_loss = torch.nn.MSELoss()

scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,patience=500)
#loader = torch.utils.data.DataLoader([intx1,intx2],batch_size = 500,shuffle = True)


def closure():
    optimizer.zero_grad()
    loss,pres,bres = pde.pdeloss(y,tdc,f,tbc,bdrydat,bw)
    loss.backward()
    optimizer.step()
    nploss = loss.detach().numpy()
    scheduler.step(nploss)
    return nploss



losslist = list()


for epoch in range(20000):
    loss = closure()
    losslist.append(loss)
    if epoch %100==0:
        print("epoch: {}, loss:{}".format(epoch,loss))




with open("results/losshist.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)



x_pts = np.linspace(0,1,50)
y_pts = np.linspace(0,1,50)

ms_x, ms_y = np.meshgrid(x_pts,y_pts)

x_pts = np.ravel(ms_x).reshape(-1,1)
t_pts = np.ravel(ms_y).reshape(-1,1)
z_pts = 0.5*np.ones([2500,1])
w_pts = 0.5*np.ones([2500,1])

collocations = np.concatenate([x_pts,t_pts,z_pts,w_pts], axis=1)

u_gt1,f = g_tr.data_gen_interior(collocations)
#u_gt1 = [np.sin(np.pi*x_col)*np.sin(np.pi*y_col) for x_col,y_col in zip(x_pts,t_pts)]
#u_gt1 = [np.exp(x_col+y_col) for x_col,y_col in zip(x_pts,t_pts)]

#u_gt = np.array(u_gt1)

ms_ugt = u_gt1.reshape(ms_x.shape)

pt_x = Variable(torch.from_numpy(x_pts).float(),requires_grad=True)
pt_t = Variable(torch.from_numpy(t_pts).float(),requires_grad=True)
pt_z = Variable(torch.from_numpy(z_pts).float(),requires_grad=True)
pt_w = Variable(torch.from_numpy(w_pts).float(),requires_grad=True)

coll = torch.cat([pt_x,pt_t,pt_z,pt_w], dim = 1)

pt_y = y(coll)
y = pt_y.data.cpu().numpy()
ms_ysol = y.reshape(ms_x.shape)




   

fig_1 = plt.figure(1, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ysol, cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('NNsolution',bbox_inches='tight')


fig_2 = plt.figure(2, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ugt, cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('GTsolution',bbox_inches='tight')

fig_3 = plt.figure(3, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,abs(ms_ugt-ms_ysol), cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('Error',bbox_inches='tight')    