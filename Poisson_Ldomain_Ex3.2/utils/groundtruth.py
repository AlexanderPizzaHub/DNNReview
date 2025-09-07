import numpy as np
from sympy import *
from . import tools

r, theta = symbols('r theta')
variables = [r,theta]

def near(x,y,tolerance):
    if Or(x-y < tolerance, y-x < tolerance):
        return True
    else:
        return False


class gt_gen(object):
    def __init__(self):
        self.variable = variables


        self.f = (2/27)*r**(-5/3)*sin((2/3)*theta)-(4/9)*r**(-1)*sin((2/3)*theta)
        self.bdry = r**(2/3) * sin((2/3) * (theta))
        self.y = r**(2/3) * sin((2/3) * (theta))
        

    def generate_by_rad(self,func,collocations_rad):
        #shape of collocation: (N,2)
        ldfunc = lambdify(variables,func,'numpy')
        data = [
        ldfunc(d[0],d[1]) for d in collocations_rad
        ]
        return np.array(data)
    
    def generate_by_cart(self,func,collocations_cart):
        #shape of collocation: (N,2)
        collocations_rad = tools.from_cart_to_rad(collocations_cart)
        return self.generate_by_rad(func,collocations_rad)
