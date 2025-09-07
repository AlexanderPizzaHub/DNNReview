import torch

mse_loss = torch.nn.MSELoss()

def pde(x1,x2,x3,x4,x5,x6,net):
    out = net(x1,x2,x3,x4,x5,x6)
    u_x1 = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_x2 = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_x3 = torch.autograd.grad(out.sum(),x3,create_graph=True)[0]
    u_x4 = torch.autograd.grad(out.sum(),x4,create_graph=True)[0]
    u_x5 = torch.autograd.grad(out.sum(),x5,create_graph=True)[0]
    u_x6 = torch.autograd.grad(out.sum(),x6,create_graph=True)[0]
    

    return u_x1,u_x2,u_x3,u_x4,u_x5,u_x6

def bdry(x1,x2,x3,x4,x5,x6,net):
    out = net(x1,x2,x3,x4,x5,x6)

    return out



def pdeloss(net,intx1,intx2,intx3,intx4,intx5,intx6,pdedata,bdx1,bdx2,bdx3,bdx4,bdx5,bdx6,bdrydata,bw):
    out = net(intx1,intx2,intx3,intx4,intx5,intx6).reshape(-1,1)
    u_x1, u_x2,u_x3,u_x4,u_x5,u_x6= pde(intx1,intx2,intx3,intx4,intx5,intx6,net)
    bout = bdry(bdx1,bdx2,bdx3,bdx4,bdx5,bdx6,net).reshape(-1,1)
    zero_vec = torch.zeros([len(intx1),1])
    loss1 = mse_loss(u_x1,zero_vec)
    loss2 = mse_loss(u_x2,zero_vec)
    loss3 = mse_loss(u_x3,zero_vec)
    loss4 = mse_loss(u_x4,zero_vec)
    loss5 = mse_loss(u_x5,zero_vec)
    loss6 = mse_loss(u_x6,zero_vec)

    loss7 = mse_loss(bout, bdrydata)
    # dataout = torch.sqrt_(out*pdedata)
    # loss3 = mse_loss(dataout,zero_vec)
    # loss = loss1 + loss2 - bw*loss3
    dataout = torch.mean(out*pdedata)
    loss = 0.5*(loss1 + loss2 + loss3 + loss4 + loss5 + loss6) - dataout + bw*loss7
    return loss

