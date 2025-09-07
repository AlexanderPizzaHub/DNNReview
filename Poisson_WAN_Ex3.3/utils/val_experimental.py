import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle as pkl
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
used for 2D plots.
'''
class plotter(object):
    def __init__(self,resolution=50):
        self.res = resolution

        val_x=np.arange(0,1,1/resolution).reshape([resolution,1])
        val_y=np.arange(0,1,1/resolution).reshape([resolution,1])

        #Generate grids to output graph
        self.val_ms_x, self.val_ms_y = np.meshgrid(val_x, val_y)
        self.plot_val_x = np.ravel(self.val_ms_x).reshape(-1,1)
        self.plot_val_y = np.ravel(self.val_ms_y).reshape(-1,1)

        #These are passed to pytorch neural networks
        self.t_val_vx = Variable(torch.from_numpy(self.plot_val_x).float()).to(device)
        self.t_val_vy = Variable(torch.from_numpy(self.plot_val_y).float()).to(device)
    '''
    one can either get collocation and compute function value by himself, or pass the function to automatically compute.
    
    '''
    def get_tensor_collocation(self):
        return [self.t_vx,self.t_vy]

    def compute_function(self,f):
        return f(self.val_x,self.val_y)


    #!!!For drawing functions, everything are two dimentional. Which kind of plot is more suitable?
    def plot_2D(self,net,path):
        pt_u = net(self.t_val_vx,self.t_val_vy).detach().numpy().reshape([50,50])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(self.val_ms_x,self.val_ms_y,pt_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig(path)
        plt.close()

    #In coupled method, control is obtained by projection
    def plot_2D_with_proj(self,net,projector,path,low,high):
        pt_u = projector(net(self.t_val_vx,self.t_val_vy)/(-alpha),low,high).detach().numpy().reshape([50,50])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(self.val_ms_x,self.val_ms_y,pt_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig(path)
        plt.close()


#primal state ground truth
u_gt_np = [
    np.sin(np.pi*val_x[ind,0])*np.sin(np.pi*val_y[ind,0]) for ind in range(len(val_x))
]

def generate_cgt(x,y,low,high):
    u = 2 * (np.pi**2) * np.sin(np.pi*x)*np.sin(np.pi*y)
    if u > high:
        u = high
    if u < low:
        u=low
    return u
    
#control ground truth
#Constraints applied.
ctrl_gt_np = [
    generate_cgt(val_x[ind,0],val_y[ind,0],-99999.0,99999.0) for ind in range(len(val_x))
]

alpha = 0.01

#adjoint state ground truth
p_gt_np = [-2*alpha*(np.pi**2)*y for y in u_gt_np]

u_gt = torch.from_numpy(np.array(u_gt_np).reshape(-1,1)).float()
ctrl_gt = torch.from_numpy(np.array(ctrl_gt_np).reshape(-1,1)).float()
p_gt = torch.from_numpy(np.array(p_gt_np).reshape(-1,1)).float()

#




mse_loss = torch.nn.MSELoss()
u_L2 = torch.sqrt(torch.mean(torch.square(u_gt)))
ctrl_L2 = torch.sqrt(torch.mean(torch.square(ctrl_gt)))




'''
In coupled method, it solves primal state and adjoint state at the same time. 
Each state contains a boundary loss and pde loss, which will be recorded.
Adding these two loss under given penalty weight, it gives the total loss, alse being recorded.
We should also evaluate the cost, although we do not use it during training.

For validation, the primal state groundtruth and contrl groundtruth will be used.
The validation error has two component: relative L2 and relative L_infinity, and validation grid is on fixed 50*50 grid.
'''
class record_couple(object):
    def __init__(self):
        self.losslist = list()
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()
        self.adjhist = list()
        self.adjres = list()
        self.adjbc = list()
        self.vhist_u = list()
        self.vhist_ctrl = list()
        self.costhist = list()
        self.epoch = 0

    def updateTL(self,loss):
        self.epoch= self.epoch+1
        self.losslist.append(loss)
        

    def updatePL(self,pl,ppde,pbc,al,apde,abc):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        self.adjhist.append(al)
        self.adjres.append(apde)
        self.adjbc.append(abc)
    
    def updateCL(self,cl):
        self.costhist.append(cl)

    def updateVL(self,vl_u,vl_p):
        self.vhist_u.append(vl_u)
        self.vhist_ctrl.append(vl_p)
    
    def validate(self,u,ctrl,projector,low,high):
        with torch.no_grad():
            vu = np.sqrt(mse_loss(u(t_vx,t_vy),u_gt).detach().numpy())/u_L2
            vc = np.sqrt(mse_loss(projector(ctrl(t_vx,t_vy)/(-alpha),low,high),ctrl_gt).detach().numpy())/ctrl_L2

        self.updateVL(vu,vc)

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.adjhist,self.vhist_u,self.vhist_ctrl]
    
    def plotinfo(self,path):
        plt.subplots(6,figsize=[30,20])
        plt.subplot(231)
        plt.loglog(self.losslist)
        plt.title("total loss")

        plt.subplot(232)
        plt.loglog(self.pdehist)
        plt.loglog(self.adjhist)
        plt.legend(['pde','adj'])
        plt.title("pde/adj loss")

        plt.subplot(233)
        plt.loglog(self.vhist_u)
        plt.loglog(self.vhist_ctrl)
        plt.legend(['state validation','control validation'])
        plt.title("validation")

        plt.subplot(234)
        plt.loglog(self.costhist)
        plt.title("cost objective")

        plt.subplot(235)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.title("primal state loss")
        plt.legend(['pde residual','boundary condition'])

        plt.subplot(236)
        plt.loglog(self.adjres)
        plt.loglog(self.adjbc)
        plt.title('adjoint state loss')
        plt.legend(['pde residual','boundary condition'])

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)


'''
In penalty method, it involves no adjoint equations. So there wil be only one boundary/pde record.
But in addition, here will be cost objective.

The validation is same as above.
'''
class record_penalty(object):
    def __init__(self):
        self.losslist = list()
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()
        self.vhist_u = list()
        self.vhist_ctrl = list()
        self.epoch = 0
        self.costhist = list()

    def updateTL(self,loss):
        self.epoch= self.epoch+1
        self.losslist.append(loss)
        

    def updatePL(self,pl,ppde,pbc,cost):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        self.costhist.append(cost)

    def updateVL(self,vl_u,vl_p):
        self.vhist_u.append(vl_u)
        self.vhist_ctrl.append(vl_p)
    
    def validate(self,u,ctrl):
        #In penalty, ctrl itself is NN and do not need projection.
        with torch.no_grad():
            vu = np.sqrt(mse_loss(u(t_vx,t_vy),u_gt).detach().numpy())/u_L2
            vc = np.sqrt(mse_loss(ctrl(t_vx,t_vy),ctrl_gt).detach().numpy())/ctrl_L2

        self.updateVL(vu,vc)
        return vu,vc

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.costhist,self.vhist_u,self.vhist_ctrl]
    
    def plotinfo(self,path):
        plt.subplots(4,figsize=[20,20])
        plt.subplot(221)
        plt.loglog(self.losslist)
        plt.title("total loss")

        plt.subplot(222)
        plt.loglog(self.pdehist)
        plt.loglog(self.costhist)
        plt.legend(['pde loss','cost objective'])
        plt.title("pde loss/cost")

        plt.subplot(223)
        plt.loglog(self.vhist_u)
        plt.loglog(self.vhist_ctrl)
        plt.title("validation")
        plt.legend(['state validation', 'control validation'])

        plt.subplot(224)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.legend(['pde residual','boundary residual'])
        plt.title("pinn loss")

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)


'''
In AONN method, primal state solve and adjoint solve are splitted. Each of them contains pde loss and boundary loss, 
which is be recorded seperately. The record is using individual epoch.

The epoch records the outer loop.

Similar to coupled method, the cost objective will not be used, but still be recorded.

Particularly, in AONN, the control NN is learned from the project GD, hence it is always admissible; there is no total loss.
'''
class record_AONN(object):
    def __init__(self):
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()

        self.adjhist = list()
        self.adjres = list()
        self.adjbc = list()

        self.vhist_u = list()
        self.vhist_ctrl = list()

        self.costhist = list()
        self.epoch = 0

    def updateEpoch(self):
        self.epoch= self.epoch+1
        

    def updatePL(self,pl,ppde,pbc):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        
    def updateAL(self,al,apde,abc):
        self.adjhist.append(al)
        self.adjres.append(apde)
        self.adjbc.append(abc)

    def updateCL(self,cl):
        self.costhist.append(cl)

    def updateVL(self,vl_u,vl_p):
        self.vhist_u.append(vl_u)
        self.vhist_ctrl.append(vl_p)
    
    def validate(self,u,ctrl):
        with torch.no_grad():
            vu = np.sqrt(mse_loss(u(t_vx,t_vy),u_gt).detach().numpy())/u_L2
            vc = np.sqrt(mse_loss(ctrl(t_vx,t_vy),ctrl_gt).detach().numpy())/ctrl_L2

        self.updateVL(vu,vc)

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.adjhist,self.vhist_u,self.vhist_ctrl]
    
    def plotinfo(self,path):
        plt.subplots(6,figsize=[30,20])

        plt.subplot(231)
        plt.loglog(self.pdehist)
        plt.title("primal state loss")

        plt.subplot(232)
        plt.loglog(self.adjhist)
        plt.title('adjoint state loss')

        plt.subplot(233)
        plt.loglog(self.vhist_u)
        plt.loglog(self.vhist_ctrl)
        plt.legend(['state validation','control validation'])
        plt.title("validation")

        plt.subplot(234)
        plt.loglog(self.costhist)
        plt.title("cost objective")

        plt.subplot(235)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.title("primal state loss")
        plt.legend(['pde residual','boundary condition'])

        plt.subplot(236)
        plt.loglog(self.adjres)
        plt.loglog(self.adjbc)
        plt.title('adjoint state loss')
        plt.legend(['pde residual','boundary condition'])

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)



'''
This is a general info recorder for the experiment.
'''
class expInfo(object):
    def __init__(self):
        self.bestValidation_u = None
        self.bestValidation_c = None
        self.bestTraining = None    #Record the best loss
        self.bestCost = None #record the best cost value
        self.epoch_Termination = None   #if we choose termination upon reaching a given accuracy, it records the number of iterations.
        self.exp_description = None #a text description of this experiment, including comments or basic settings
        self.walltime = None #records the running time

    def saveinfo(self,path):
        info = json.dumps(self.__dict__,indent=4,separators=(',',':'))
        f = open(path,'w')
        f.write(info)
        f.close()
        return True
    