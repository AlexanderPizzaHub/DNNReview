import pickle as pkl
from scipy.stats import uniform
import numpy as np
import os
from utils import groundtruth as gt


N = 10000
dataname = '10000pts'



domain_data_x = uniform.rvs(size=[N,6])

domain_data = np.array(domain_data_x)
print(domain_data.shape)


Nb = 2000

def generate_random_bdry(Nb):
    bdry_col = uniform.rvs(size=Nb*6).reshape([Nb,6])
    for i in range(Nb):
        randind = np.random.randint(0,6)
        if bdry_col[i,randind] <= 0.5:
            bdry_col[i,randind] = 0.0
        else:
            bdry_col[i,randind] = 1.0

    return bdry_col

bdry_col = generate_random_bdry(Nb)
'''
Nb = 160
Nb_line = 41
#x=0~1, y=0
bx_0 = np.array(np.linspace(0.,1.,Nb_line)).reshape([Nb_line,1])
by_0 = np.zeros([Nb_line,1])

#x=0~1, y=1
bx_1 = np.array(np.linspace(0.,1.,Nb_line)).reshape([Nb_line,1])
by_1 = np.zeros([Nb_line,1]) + 1.0

#x=0, y=0~1
bx_2 = np.zeros([Nb_line,1])
by_2 = np.array(np.linspace(0.,1.,Nb_line)).reshape([Nb_line,1])

#x=1, y=0~1
bx_3 = np.zeros([Nb_line,1]) + 1.0
by_3 = np.array(np.linspace(0.,1.,Nb_line)).reshape([Nb_line,1])



bdry_col_0 = np.concatenate([bx_0,by_0],axis=1)
bdry_col_1 = np.concatenate([bx_1,by_1],axis=1)
bdry_col_2 = np.concatenate([bx_2,by_2],axis=1)
bdry_col_3 = np.concatenate([bx_3,by_3],axis=1)'''

#bdry_col = np.unique(np.concatenate([bdry_col_0,bdry_col_1,bdry_col_2,bdry_col_3]),axis=0)
print(bdry_col.shape)




if not os.path.exists('dataset/'):
    os.makedirs('dataset/')
with open('dataset/'+dataname,'wb') as pfile:
    pkl.dump(domain_data,pfile)
    pkl.dump(bdry_col,pfile)


gtgen = gt.gt_gen()

#ygt,fgt = gt.data_gen_interior(domain_data)
#bdry_dat = gt.data_gen_bdry(bdry_col)
ygt = gtgen.generate_data(gtgen.y,domain_data)
fgt = gtgen.generate_data(gtgen.f,domain_data)
bdry_dat = gtgen.generate_data(gtgen.y,bdry_col)



with open("dataset/gt_on_{}".format(dataname),'wb') as pfile:
    pkl.dump(ygt,pfile)
    pkl.dump(fgt,pfile)
    pkl.dump(bdry_dat,pfile)

   