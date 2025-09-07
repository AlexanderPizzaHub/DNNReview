import torch

#Function to compute pde term

mse_loss = torch.nn.MSELoss()
#use x^2 to verify the laplacian
def pde(x,y,net):
    out = net(x,y)

    u_x = torch.autograd.grad(out.sum(),x,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]

    u_y = torch.autograd.grad(out.sum(),y,create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(),y,create_graph=True)[0]

    return -u_xx - u_yy 

def pde_misfit(x,y,net):
    out = net(x,y)

    u_x = torch.autograd.grad(out.sum(),x,create_graph=True,retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=False,retain_graph=True)[0]

    u_y = torch.autograd.grad(out.sum(),y,create_graph=True,retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(),y,create_graph=False,retain_graph=True)[0]

    return -u_xx - u_yy 

def adjoint(x,y,net):
    out = net(x,y)

    u_x = torch.autograd.grad(out.sum(),x,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]

    u_y = torch.autograd.grad(out.sum(),y,create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(),y,create_graph=True)[0]

    return -u_xx - u_yy

#Function to compute the bdry
def bdry(bx,by,net):
    out = net(bx,by)
    return out


#The loss
def pdeloss(net,px,py,pdedata,bx,by,bdrydata,bw_in=None):
    
    #pdedata is f.
    pout = pde(px,py,net)
    
    bout = bdry(bx,by,net)
    
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    
    bw = bw_in
    
    loss = (1.0-bw)*pres + bw*bres

    return loss,pres,bres,pdedata-pout,bout-bdrydata

def adjloss(net,px,py,pdedata,bx,by,bdrydata,bw_in=None):
    
    #pdedata is f.
    pout = adjoint(px,py,net)
    
    bout = bdry(bx,by,net)
    
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    bw = bw_in
    loss = (1.0-bw)*pres + bw*bres
    #loss = pres + bw*bres
    return loss,pres,bres

def costfunc(y,data,u,ld,cx,cy,nx,ny):
    #evaluate by trapezoidal

    yout = y(cx,cy)
  
    yre = yout.reshape([nx,ny])
    
    if not isinstance(u,torch.Tensor):
        uout = u(cx,cy)
    else:
        uout = u
    ure = uout.reshape([nx,ny])

    dre = data.reshape([nx,ny])
    
    misfit = 0.5 *torch.square(yre-dre) + ld * 0.5 * torch.square(ure)

    cost = torch.trapezoid(
        torch.trapezoid(misfit,dx=1/(nx-1)),
        dx=1/(ny-1)
        )
    
    return cost

def cost_mse(y,data,u,ld,cx,cy):
    yout = y(cx,cy)
    if not isinstance(u,torch.Tensor):
        uout = u(cx,cy)
    else:
        uout = u
    misfit = 0.5 *torch.square(yout-data) + ld * 0.5 * torch.square(uout)
    cost = torch.mean(misfit)
    return cost
