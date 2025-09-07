import torch

mse_loss = torch.nn.MSELoss()

def doublegrad(y,x):
    y_x = torch.autograd.grad(y.sum(),x,create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x.sum(),x,create_graph=True)[0]
    
    return y_xx


def pde(x,net):
    x_block = torch.stack(x,dim=1)
    out = net(x_block)

    laplace_list = []
    for dim in range(x_block.shape[-1]):
        laplace_list.append(doublegrad(out,x[dim]))

    laplacian = torch.stack(laplace_list,dim=0).sum(dim=0)
    return -laplacian


    #Function to compute the bdry
def bdry(bx,net):
    bx_block = torch.stack(bx,dim=1)
    out = net(bx_block)
    return out


#The loss
def pdeloss(net,px,pdedata,bx,bdrydata,bw):
    
    pout = pde(px,net).unsqueeze(dim=-1)
    bout = bdry(bx,net)
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    
    loss = pres + bw*bres

    return loss,pres,bres

