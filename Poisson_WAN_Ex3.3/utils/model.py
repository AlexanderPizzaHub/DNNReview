import torch
import numpy as np

class solution(torch.nn.Module):
    def __init__(self):
        super(solution,self).__init__()
        self.input_layer = torch.nn.Linear(2,30)
        self.Hidden1 = torch.nn.Linear(30,30) 
        self.Hidden2 = torch.nn.Linear(30,30)
        self.Hidden3 = torch.nn.Linear(30,30)
        self.output_layer = torch.nn.Linear(30,1)


    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)

        x1 = torch.tanh(self.input_layer(inputs))

        x2 = torch.tanh(self.Hidden1(x1)) + x1
        x2 = torch.tanh(self.Hidden2(x2)) 
        x3 = torch.tanh(self.Hidden3(x2)) + x2

        x3 = self.output_layer(x3)

        return x3

class adverse(torch.nn.Module):
    def __init__(self):
        super(adverse,self).__init__()
        self.input_layer = torch.nn.Linear(2,30)
        self.Hidden1 = torch.nn.Linear(30,30) 
        self.Hidden2 = torch.nn.Linear(30,30)
        self.Hidden3 = torch.nn.Linear(30,30)
        self.output_layer = torch.nn.Linear(30,1)


    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)

        x1 = torch.tanh(self.input_layer(inputs))

        x2 = torch.tanh(self.Hidden1(x1)) + x1
        x2 = torch.tanh(self.Hidden2(x2)) 
        x3 = torch.tanh(self.Hidden3(x2)) + x2

        x3 = self.output_layer(x3)

        out = x3 * torch.sin(torch.pi*x) * torch.sin(torch.pi*y)

        return out


class ctrl(torch.nn.Module):
    def __init__(self):
        super(ctrl,self).__init__()
        self.input_layer = torch.nn.Linear(2,30)
        self.H1 = torch.nn.Linear(30,30) 
        self.H2 = torch.nn.Linear(30,30)
        self.output_layer = torch.nn.Linear(30,1)
        
    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.tanh(self.input_layer(inputs))

        x2 = torch.tanh(self.H1(x1)) 
        x3 = torch.tanh(self.H2(x2))
        x3 = self.output_layer(x3)

        return x3
    

    #Define NN models. Maps from R2 -> R1.
class NNsol(torch.nn.Module):
    def __init__(self):
        super(NNsol,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L3 = torch.nn.Linear(30,30)
        self.L4 = torch.nn.Linear(30,30)
        self.L5 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.sigmoid(self.L1(inputs))

        x2 = torch.sigmoid(self.L2(x1)) 
        x2 = torch.sigmoid(self.L3(x2)) + x1

        x3 = torch.sigmoid(self.L4(x2)) + x2
        x3 = self.L5(x3)

        return x3

class NNctrl(torch.nn.Module):
    def __init__(self):
        super(NNctrl,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L3 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.sigmoid(self.L1(inputs))

        x2 = torch.sigmoid(self.L2(x1)) 
        x2 = torch.sigmoid(self.L3(x2))

        return x2
    
class NNsol_noskip(torch.nn.Module):
    def __init__(self):
        super(NNsol_noskip,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L3 = torch.nn.Linear(30,30)
        self.L4 = torch.nn.Linear(30,30)
        self.L5 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.tanh(self.L1(inputs))

        x2 = torch.tanh(self.L2(x1)) 
        x2 = torch.tanh(self.L3(x2))

        x3 = torch.tanh(self.L4(x2))
        x3 = self.L5(x3)

        return x3

class NNsol_noskip_large(torch.nn.Module):
    def __init__(self):
        super(NNsol_noskip_large,self).__init__()
        self.L1 = torch.nn.Linear(2,100)
        self.L2 = torch.nn.Linear(100,100) 
        self.L3 = torch.nn.Linear(100,100)
        self.L4 = torch.nn.Linear(100,100)
        self.L5 = torch.nn.Linear(100,100)
        self.L6 = torch.nn.Linear(100,100)
        self.L7 = torch.nn.Linear(100,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.tanh(self.L1(inputs))

        x2 = torch.tanh(self.L2(x1)) 
        x2 = torch.tanh(self.L3(x2))
        x3 = torch.tanh(self.L4(x2))
        x4 = torch.tanh(self.L5(x3))
        x5 = torch.tanh(self.L6(x4))

        x6 = self.L7(x5)

        return x6

silu = torch.nn.SiLU()
class sol_swish(torch.nn.Module):
    def __init__(self):
        super(sol_swish,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L3 = torch.nn.Linear(30,30)
        self.L4 = torch.nn.Linear(30,30)
        self.L5 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = silu(self.L1(inputs))

        x2 = silu(self.L2(x1)) 
        x2 = silu(self.L3(x2)) + x1

        x3 = silu(self.L4(x2)) + x2
        x3 = self.L5(x3)

        return x3

class ctrl_swish(torch.nn.Module):
    def __init__(self):
        super(ctrl_swish,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L3 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = silu(self.L1(inputs))

        x2 = silu(self.L2(x1)) 
        x2 = silu(self.L3(x2))

        return x2
    

def projection_softmax(net_values,low,high,a,hard_constraint=False):
    m = (high+low)/2.
    delta = high - m
    if not hard_constraint:
        b = 2*delta*(1+np.exp(-a*delta))/(1-np.exp(-a*delta))
        sig_x = torch.sigmoid(a*(net_values-m)) - 0.5
        out = b*sig_x + m
    else:
        b = 2*(high-m)
        sig_x = torch.sigmoid(a*(net_values-m)) - 0.5
        out = b*sig_x + m

    return out

def projection_clamp(net_values,low,high):
    out = torch.clamp(net_values,low,high)

    return out


def init_weights(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    