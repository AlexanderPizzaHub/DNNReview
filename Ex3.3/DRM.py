from time import time
from tracemalloc import start
import numpy as np
import torch
import torch.optim as opt
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import model,pde,data,tools,g_tr,validation
from utils import model,tools,groundtruth,validation
from utils import pde_DRM as pde
from time import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float64)

exp_name = 'DRM_t5'

settings = {
        'dataname':'5000pts',
        'interior_data':5000,
        'boundary_data':1000,
        
        'max_iter': 5000,
        'boundary_weight': 100,
        'lr': 1e-4,
        'NN_arc': [2,30,30,30,30,1],
        'optimizer': 'Adam',
        'scheduler': 'MultistepLr',
        #'scheduler_params': {'milestones': [3000, 4000], 'gamma': 0.1},
        'scheduler_params': {'milestones': [7000, 9000], 'gamma': 0.1},
        'datatype': 'float64',
}

dataname =settings['dataname']
name = 'results/' + exp_name +"/"
bw = settings['boundary_weight']

y = model.NN()
y.apply(model.init_weights)

dataname = settings['dataname']
#name = 'results/'

bw = settings['boundary_weight']

if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"y_plot/"):
    os.makedirs(name+"y_plot/")


params = list(y.parameters())

with open("dataset/"+dataname,'rb') as pfile:
    int_col = pkl.load(pfile)
    bdry_col = pkl.load(pfile)

print(int_col.shape,bdry_col.shape)

intx1,intx2 = np.split(int_col,2,axis=1)
bdx1,bdx2 = np.split(bdry_col,2,axis=1)

tintx1,tintx2,tbdx1,tbdx2 = tools.from_numpy_to_tensor([intx1,intx2,bdx1,bdx2],[True,True,False,False,],dtype=torch.float64)


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdry_np = pkl.load(pfile)

f,bdrydat,ygt = tools.from_numpy_to_tensor([f_np,bdry_np,y_gt],[False,False,False],dtype=torch.float64)

bdrydat = bdrydat.reshape(-1,1)
f = f.reshape(-1,1)


optimizer = opt.Adam(params,lr=1e-4)

mse_loss = torch.nn.MSELoss()

#scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,patience=500)
#loader = torch.utils.data.DataLoader([intx1,intx2],batch_size = 500,shuffle = True)
scheduler = opt.lr_scheduler.MultiStepLR(optimizer,**settings['scheduler_params'])


def closure():
    optimizer.zero_grad()
    loss = pde.pdeloss(y,tintx1,tintx2,f,tbdx1,tbdx2,bdrydat,bw)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach().numpy()

losslist = list()


for epoch in range(settings['max_iter']):
    loss = closure()
    losslist.append(loss)
    if epoch %100==0:
        vy,_,_ = validation.validate(y)
        validation.plot_2D(y,name+"y_plot/"+'y'.format(epoch))
        torch.save(y,name+'y.pt')
        print("epoch: {}, loss:{}, val:{}".format(epoch,loss,vy))

with open("results/losshist.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)

torch.save(y,name+'y.pt')

settings['rel_error_L2'] = float(vy) 

settings = {
    exp_name:settings
}

validation.plot_2D(y,name+"y_plot/"+'final')
rec = tools.recorder('exp_table.json')
rec.write(settings)
rec.save()




'''

# plotting solutions
domain_data_x = np.linspace(-1,1,100)
domain_data_y = np.linspace(-1,1,100)

ms_x, ms_y = np.meshgrid(domain_data_x,domain_data_y)

x_pts = np.ravel(ms_x).reshape(-1,1)
y_pts = np.ravel(ms_y).reshape(-1,1)
collocations = np.concatenate([x_pts,y_pts], axis=1)'''

'''
#use polar coord for ground truth solution
domain_col_x, domain_col_y = collocations[:,0], collocations[:,1]
domain_col_r = np.sqrt(domain_col_x**2 + domain_col_y**2)
domain_col_theta = np.arctan2(domain_col_y,domain_col_x)
theta_negative = domain_col_theta < 0
domain_col_theta[theta_negative] += 2*np.pi 
domain_col_pol = np.column_stack((domain_col_r,domain_col_theta))'''


'''
u_gt, f_blah = g_tr.data_gen_interior(domain_col_pol)
ms_ugt = u_gt.reshape(ms_x.shape)

pt_x = Variable(torch.from_numpy(x_pts).float(),requires_grad=True)
pt_y = Variable(torch.from_numpy(y_pts).float(),requires_grad=True)

y_val = y(pt_x,pt_y)
y_val = y_val.data.cpu().numpy()
ms_ysol = y_val.reshape(ms_x.shape)

mark = np.logical_and(ms_x>0.0,ms_y<0)
ms_ysol[mark] = np.nan
ms_ugt[mark] = np.nan 
abs_err = np.abs(ms_ugt-ms_ysol)
abs_err[mark] = np.nan



##Error computation
# Flatten the absolute error array for easier computation
abs_err_flat = abs_err[~np.isnan(abs_err)].flatten()

# Compute the mean square L2 error
mean_square_l2_error = np.mean(abs_err_flat**2)

# Compute the relative L2 error
relative_l2_error = np.sqrt(mean_square_l2_error) / np.sqrt(np.nanmean(ms_ugt**2))

print(f"Mean Square L2 Error: {mean_square_l2_error}")
print(f"Relative L2 Error: {relative_l2_error}")






fig_1 = plt.figure(1, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ysol, cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('NNsolution',bbox_inches='tight')


fig_2 = plt.figure(2, figsize=(6, 5))
plt.pcolor(ms_x, ms_y,ms_ugt, cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('GTsolution',bbox_inches='tight')



fig_3 = plt.figure(3, figsize=(6, 5))
plt.pcolor(ms_x, ms_y,abs(ms_ugt-ms_ysol), cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('Error',bbox_inches='tight')'''







