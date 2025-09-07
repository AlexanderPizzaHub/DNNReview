import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle as pkl
import json
from . import groundtruth as gt
from . import tools

dtype = torch.float64


resolution = 128
val_x1=np.arange(-1,1,2/resolution).reshape(-1,1)
val_x2=np.arange(-1,1,2/resolution).reshape(-1,1)
t_vx1 = Variable(torch.from_numpy(val_x1)).type(dtype)
t_vx2 = Variable(torch.from_numpy(val_x2)).type(dtype)



#Generate grids to output graph
val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)

plot_val_x1 = np.ravel(val_ms_x1).reshape(-1,1)
plot_val_x2 = np.ravel(val_ms_x2).reshape(-1,1)

mark =np.logical_not(np.logical_and(plot_val_x1>0,plot_val_x2<0))

plot_val_x1 = plot_val_x1[mark].reshape(-1,1)
plot_val_x2 = plot_val_x2[mark].reshape(-1,1)

t_val_vx1,t_val_vx2 = tools.from_numpy_to_tensor([plot_val_x1,plot_val_x2],[False,False],dtype=dtype)

#y_gt,f = gt.data_gen_interior(np.concatenate([plot_val_x1,plot_val_x2],axis=1))

#t_ygt = tools.from_numpy_to_tensor([y_gt],[False,False,False,False],dtype=dtype)


'''def plot_2D(net,path):
    data = net(t_val_vx1,t_val_vx2).detach().numpy().reshape([resolution,resolution])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(val_ms_x1,val_ms_x2,data, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(path)
    plt.close()'''

def plot_2D(net,path):
    data = net(t_val_vx1,t_val_vx2).detach().numpy()
    plt.scatter(plot_val_x1,plot_val_x2,c=data)
    plt.colorbar()

    plt.savefig(path)
    plt.close()

def plot_2D_vpinn(net,path):
    #data = net(torch.cat([t_val_vx1,t_val_vx2],axis=1).unsqueeze(1)).detach().numpy().reshape([resolution,resolution])
    data = net(torch.cat([t_val_vx1,t_val_vx2],axis=1).unsqueeze(1)).detach().numpy()
    plt.scatter(plot_val_x1,plot_val_x2,c=data)
    plt.colorbar()

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #surf = ax.plot_surface(val_ms_x1,val_ms_x2,data, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(path)
    plt.close()


gt_gen = gt.gt_gen()
y_gt = gt_gen.generate_by_cart(gt_gen.y,np.concatenate([plot_val_x1,plot_val_x2],axis=1)).reshape(-1,1)
y_L2 = np.sqrt(np.mean(np.square(y_gt)))#np.norm(ygt)
print("y_L2:",y_L2)

def validate(net):
    with torch.no_grad():
        y_pred = net(t_val_vx1,t_val_vx2).numpy().reshape(-1,1)
    vy = np.sqrt(np.mean(np.square(y_pred-y_gt)))/y_L2
    return vy,y_pred,y_gt

def validate_vpinn(net):
    with torch.no_grad():
        y_pred = net(torch.cat([t_val_vx1,t_val_vx2],axis=1).unsqueeze(1)).numpy().reshape(-1,1)

    vy = np.sqrt(np.mean(np.square(y_pred-y_gt)))/y_L2
    return vy,y_pred,y_gt

    