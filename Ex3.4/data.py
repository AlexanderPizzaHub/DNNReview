import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
import g_tr as gt
from sympy import *

N = 10000
dataname = '10000pts'

x1,x2,x3,x4 = symbols('x1 x2 x3 x4')
variables = [x1,x2,x3,x4]
alpha = 0.01

domain_data_list = list()
for _ in variables:
    domain_data_list.append(uniform.rvs(size=N))

domain_data = np.array(domain_data_list).T
print(domain_data.shape)




def sample_one_bdry():
    datapoint = list()
    for dim_ind in range(4):
        datapoint.append(uniform.rvs())

    dim_rand = np.random.randint(0,4)
    dim_rand_data = float(np.random.randint(2))
    datapoint[dim_rand] = dim_rand_data 
    
    return datapoint



Nb = 3000
boundary_data_list = list()
for i in range(Nb):
    boundary_data_list.append(sample_one_bdry())

bdry_col = np.array(boundary_data_list)

print(bdry_col.shape)
    
    




if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)


ygt,fgt = gt.data_gen_interior(domain_data)
bdry_dat = gt.data_gen_bdry(bdry_col)





with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(ygt,pfile)
    pkl.dump(fgt,pfile)
    pkl.dump(bdry_dat,pfile)

   