import numpy as np
from sympy import *
from . import tools

x1,x2,x3,x4,x5,x6 = symbols('x1 x2 x3 x4 x5 x6')
variables = [x1,x2,x3,x4,x5,x6]

def near(x,y,tolerance):
    if Or(x-y < tolerance, y-x < tolerance):
        return True
    else:
        return False


class gt_gen(object):
    def __init__(self):
        self.variable = variables

        #self.f = 4.0 * pi**2 * sin(pi*x1) * sin(pi*x2) * sin(pi*x3) * sin(pi*x4) * x5 * (1-x5) * x6 * (1-x6)
        #self.y = sin(pi*x1)*sin(pi*x2) * sin(pi*x3) * sin(pi*x4) * sin(pi*x5) * sin(pi*x6)
        self.y = x1*x2 + x3*x4 + x5*x6
        self.bdry = self.y 

        laplacian_y = 0
        for x in variables:
            laplacian_y += diff(self.y,x,x)

        self.f = -laplacian_y
        #self.f = (2/27)*r**(-5/3)*sin((2/3)*theta)-(4/9)*r**(-1)*sin((2/3)*theta)
    
    def generate_data(self,func,col):
        #shape of collocation: (N,2)
        ldfunc = lambdify(variables,func,'numpy')
        data = [
        ldfunc(d[0],d[1],d[2],d[3],d[4],d[5]) for d in col
        ]
        return np.array(data)
