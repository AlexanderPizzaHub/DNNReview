import torch
from torch.autograd import Variable

#This version allows training multiple neural networks withing PDE.
#gpu version still debugging

def from_numpy_to_tensor(numpys,require_grads,dtype=torch.float64,device='cpu'):
    #numpys: a list of numpy arrays.
    #requires_grads: a list of boolean to indicate whether give gradients
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=require_grads[ind]).type(dtype).to(device)
        )

    return outputs



def checkgrad(listoftensors):
    for ts in listoftensors:
        if ts.grad is not None:
            ts.grad.zero_()
    
