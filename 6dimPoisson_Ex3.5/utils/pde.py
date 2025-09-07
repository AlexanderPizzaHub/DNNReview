import torch

mse_loss = torch.nn.MSELoss()

def pde(x1,x2,x3,x4,x5,x6,net):
    out = net(x1,x2,x3,x4,x5,x6)
    u_x = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x1, create_graph=True)[0]

    u_x2 = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_x2x2 = torch.autograd.grad(u_x2.sum(),x2,create_graph=True)[0]

    u_x3 = torch.autograd.grad(out.sum(),x3,create_graph=True)[0]
    u_x3x3 = torch.autograd.grad(u_x3.sum(),x3,create_graph=True)[0]

    u_x4 = torch.autograd.grad(out.sum(),x4,create_graph=True)[0]
    u_x4x4 = torch.autograd.grad(u_x4.sum(),x4,create_graph=True)[0]

    u_x5 = torch.autograd.grad(out.sum(),x5,create_graph=True)[0]
    u_x5x5 = torch.autograd.grad(u_x5.sum(),x5,create_graph=True)[0]

    u_x6 = torch.autograd.grad(out.sum(),x6,create_graph=True)[0]
    u_x6x6 = torch.autograd.grad(u_x6.sum(),x6,create_graph=True)[0]

    return -u_xx-u_x2x2-u_x3x3-u_x4x4-u_x5x5-u_x6x6


def bdry(x1,x2,x3,x4,x5,x6,net):
    out = net(x1,x2,x3,x4,x5,x6)
    return out


def pdeloss(net,intx1,intx2,intx3,intx4,intx5,intx6,pdedata,bdx1,bdx2,bdx3,bdx4,bdx5,bdx6,bdrydata,bw):
    pout = pde(intx1,intx2,intx3,intx4,intx5,intx6,net)
    bout = bdry(bdx1,bdx2,bdx3,bdx4,bdx5,bdx6,net)
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)

    loss = pres + bw*bres

    return loss, pres, bres 

