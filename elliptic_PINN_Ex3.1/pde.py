import torch

mse_loss = torch.nn.MSELoss()

def nabla(x1,x2,out):
    u_x = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_y = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]

    return torch.concatenate([u_x, u_y],dim=1)

def div(x1,x2,out):
    #out:(N,d), where d the space dimension
    u_x = torch.autograd.grad(out[:,0].sum(),x1,create_graph=True)[0]
    u_y = torch.autograd.grad(out[:,1].sum(),x2,create_graph=True)[0]
    return u_x + u_y

def pde(x1,x2,A,beta,gamma,net):
    #A:(d,d), where d the space dimension
    # div(A \nabla(u)) + beta \dot \nabla(u) + \gamma u  
    out = net(x1,x2)
    gd = nabla(x1,x2,out) # (N,d)
    d = div(x1,x2,torch.matmul(A,gd.t()).t())

    lhs = d + torch.matmul(gd,beta) + gamma*out

    return lhs


def bdry(x1,x2,net):
    out = net(x1,x2)
    return out


def pdeloss(net,intx1,intx2,pdedata,bdx1,bdx2,bdrydata,A,beta,gamma,bw):
    pout = pde(intx1,intx2,A,beta,gamma,net)
    bout = bdry(bdx1,bdx2,net)
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)

    loss = pres + bw*bres

    return loss, pres, bres 

