import torch

mse_loss = torch.nn.MSELoss()

def pde(x1,x2,net):
    out = net(x1,x2)
    u_x = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_y = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    

    return u_x, u_y

def bdry(x1,x2,net):
    out = net(x1,x2)

    return out



def pdeloss(net,intx1,intx2,pdedata,bdx1,bdx2,bdrydata,bw):
    out = net(intx1,intx2)
    u_x, u_y = pde(intx1,intx2,net)
    bout = bdry(bdx1,bdx2,net)
    zero_vec = torch.zeros([len(intx1),1])
    loss1 = mse_loss(u_x,zero_vec)
    loss2 = mse_loss(u_y,zero_vec)
    loss3 = mse_loss(bout, bdrydata)
    # dataout = torch.sqrt_(out*pdedata)
    # loss3 = mse_loss(dataout,zero_vec)
    # loss = loss1 + loss2 - bw*loss3
    dataout = torch.mean(out*pdedata)
    loss = 0.5*loss1 + 0.5*loss2 - dataout + bw*loss3
    return loss

