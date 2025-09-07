import torch
from torch.autograd import Variable

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = 'cpu'

#This version allows training multiple neural networks withing PDE.
#gpu version still debugging

def from_numpy_to_tensor_with_grad(numpys,device=device):
    #numpys: a list of numpy arrays.
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]).float(),requires_grad=True).to(device)
        )

    return outputs

def from_numpy_to_tensor(numpys,device=device):
    #numpys: a list of numpy arrays.
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]).float(),requires_grad=False).to(device)
        )

    return outputs

    
