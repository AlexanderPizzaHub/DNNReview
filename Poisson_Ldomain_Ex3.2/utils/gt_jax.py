import numpy as np
from sympy import *
import jax.numpy as jnp
#from . import tools

r, theta = symbols('r theta')
variables = [r,theta]

def near(x,y,tolerance):
    if Or(x-y < tolerance, y-x < tolerance):
        return True
    else:
        return False

def from_cart_to_rad(cart_col):
    
    x, y = cart_col[:,0], cart_col[:,1]
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y,x)
    #tmp = np.arange(0,len(theta))
    theta_negative = theta<0
    #print(theta_negative)
    
    theta = theta.at[theta_negative].add(2*jnp.pi)
    #rad_col = np.column_stack((r,theta))
    return jnp.array([r,theta]).T

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
        return jnp.array(data)
    
    def generate_by_cart(self,func,collocations_cart):
        #shape of collocation: (N,2)
        collocations_rad = from_cart_to_rad(collocations_cart)
        return self.generate_by_rad(func,collocations_rad)
