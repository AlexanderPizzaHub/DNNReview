import numpy as np
from sympy import *


# div(A nabla(y)) + beta dot nabla(y) + gamma y = f

x1,x2 = symbols('x1 x2')
#y = exp(x1*(1-x1))*sin(pi*x2) + exp(x2*(1-x2))*sin(pi*x1)
y = sin(pi*x1)*sin(pi*x2)

A = Matrix([[4.,-1.],[-1.,3.]])
b = Matrix([[1.],[2.]])
g = 2.

variables = [x1,x2]

nabla_y = Matrix([y]).jacobian(variables).T

A_nabla_y = A*nabla_y #(2,1)

div_A_nabla_y = A_nabla_y.row(0).diff(x1) + A_nabla_y.row(1).diff(x2)



f = div_A_nabla_y + nabla_y.T*b + g * Matrix([[y]])

print(f)

#Generates all the ground truth data needed. 

#boundary is periodic, not needed.
ldy = lambdify(variables,y,'numpy')
ldf = lambdify(variables,f,'numpy')

def from_seq_to_array(items):
    out = list()
    for item in items:
        out.append(np.array(item).reshape(-1,1))
    
    if len(out)==1:
        out = out[0]
    return out

def data_gen_interior(collocations):
    #how to parse the input?
    y_gt = [ldy(d[0],d[1]) for d in collocations]
        
    
    f = [ldf(d[0],d[1]) for d in collocations]

    return from_seq_to_array([y_gt,f])

def data_gen_bdry(collocations):
    #how to parse the input?
    y_gtbdry = [ldy(d[0],d[1]) for d in collocations]

    return from_seq_to_array([y_gtbdry])