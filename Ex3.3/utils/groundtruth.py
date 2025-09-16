import numpy as np
from sympy import *
from . import tools

x1,x2 = symbols('x1 x2')
variables = [x1,x2]

def near(x,y,tolerance):
    if Or(x-y < tolerance, y-x < tolerance):
        return True
    else:
        return False


class gt_gen(object):
    def __init__(self):
        self.variable = variables

        self.f = 2.0 
        self.y = Min(x1**2,(1-x1)**2)
        self.bdry = Min(x1**2,(1-x1)**2)

        #self.f = (2/27)*r**(-5/3)*sin((2/3)*theta)-(4/9)*r**(-1)*sin((2/3)*theta)
    
    def generate_data(self,func,col):
        #shape of collocation: (N,2)
        ldfunc = lambdify(variables,func,'numpy')
        data = [
        ldfunc(d[0],d[1]) for d in col
        ]
        return np.array(data)
