import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
from utils import tools
from utils import groundtruth as gt

N = 5000
dataname = '5000pts'

def sample_one_point():
    '''
    Sample one point from the domain.
    '''
    x = 2*uniform.rvs()-1.
    y = 2*uniform.rvs()-1.
    return x,y


def gen_domain(N):
    '''
    Generate domain points.
    '''
    domain_col = list()
    while len(domain_col)<N:
        x,y = sample_one_point()
        if not (x>0. and y<0.):
            domain_col.append([x,y])
    return np.array(domain_col)


domain_data = gen_domain(N)
print(domain_data.shape)


Nb = 1000
''' 
6 edges in total:

1: x:[0,1], y=0     --> length=1
2: x=1,     y:[0,1] --> length=1
3: x:[-1,1],y=1     --> length=2
4: x=-1,    y:[-1,1]--> length=2
5: x:[-1,0],y=-1    --> length=1
6: x=0,     y:[-1,0]--> length=1
'''

def sample_one_bdry():
    '''
    Sample one point from the boundary.
    '''
    edge_ind = np.random.randint(0,8)
    if edge_ind == 0:
        x = uniform.rvs()
        y = 0.

    elif edge_ind == 1:
        x = 1.
        y = uniform.rvs()
    elif (edge_ind == 2 or edge_ind == 3):
        x = uniform.rvs()*2-1.
        y = 1.
    elif (edge_ind == 4 or edge_ind == 5):
        x = -1.
        y = uniform.rvs()*2-1.
    elif edge_ind == 6:
        x = uniform.rvs() - 1. 
        y = -1. 
    elif edge_ind == 7:
        x = 0. 
        y = uniform.rvs() - 1.
    else:
        raise ValueError('edge_ind out of range!')

    return [x,y]

def gen_bd(Nb):
    b_col = list()
    for _ in range(Nb):
        b_col.append(sample_one_bdry())

    return np.array(b_col)


bdry_col = gen_bd(Nb)
print(bdry_col.shape)


if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)


#domain_col_pol = tools.from_cart_to_rad(domain_data)
#bdry_col_pol = tools.from_cart_to_rad(bdry_col)

#ygt,fgt = gt.data_gen_interior(domain_col_pol)
#bdry_dat = gt.data_gen_bdry(bdry_col_pol)
dg = gt.gt_gen()
ygt = dg.generate_by_cart(dg.y,domain_data)
fgt = dg.generate_by_cart(dg.f,domain_data)
bdry_dat = dg.generate_by_cart(dg.bdry,bdry_col)


with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(ygt,pfile)
    pkl.dump(fgt,pfile)
    pkl.dump(bdry_dat,pfile)

   