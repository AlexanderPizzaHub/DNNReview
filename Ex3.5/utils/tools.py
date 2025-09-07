import torch
from torch.autograd import Variable
import numpy as np
import json
import os,io
import gmsh


def from_numpy_to_tensor_with_grad(numpys,dtype = torch.float64):
    #numpys: a list of numpy arrays.
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]).float(),requires_grad=True).type(dtype)
        )

    return outputs

def from_numpy_to_tensor(numpys,require_grads,dtype=torch.float64):
    #numpys: a list of numpy arrays.
    #requires_grads: a list of boolean to indicate whether give gradients
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=require_grads[ind]).type(dtype)
        )

    return outputs

def near(a,b,tol=1e-8):
    if np.abs(a-b)<tol:
        return True
    else:
        return False

def from_cart_to_rad(cart_col):
    x, y = cart_col[:,0], cart_col[:,1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    theta_negative = theta < 0
    
    theta[theta_negative] += 2*np.pi 
    #rad_col = np.column_stack((r,theta))
    return np.array([r,theta]).T


def load_mesh(path):
    gmsh.initialize()
    gmsh.open(path)
    dim = -1 
    tag = -1 

    nodeTags, nodeCoords, parametricCoord = gmsh.model.mesh.getNodes(dim, tag, False, False)
    coords = nodeCoords.reshape((-1, 3))
    defined_element_type = gmsh.model.mesh.getElementTypes(dim, tag)
    
    eletype= 2 #triangle surface elements. type 1 -> line elements. type 15 -> point elements.
    #currently the boundary has to be mannually identified. -> check node position and identify.

    elementTags, nodeTags = gmsh.model.mesh.getElementsByType(eletype,tag)
    nodeTags = nodeTags.reshape((-1, 3)).astype(np.int32)
    surface_area = 0. 
    elem_area = []
    for element in nodeTags:
        node1 = coords[int(element[0]-1),:]
        node2 = coords[int(element[1]-1),:]
        node3 = coords[int(element[2]-1),:]
        area = 0.5 * abs(node1[0]*(node2[1]-node3[1]) + node2[0]*(node3[1]-node1[1]) + node3[0]*(node1[1]-node2[1]))
        #surface_area += area
        elem_area.append(area)

    gmsh.finalize()
    return coords[:,:-1],nodeTags,elem_area


class mesh(object):
    def __init__(self,coord,elem):
        self.crd = coord
        self.em = elem - elem.min()
        self.em_flattened = torch.tensor(self.em).reshape(-1)
        
        self.nrElem = len(elem)
        self.nrNode = len(coord)

        self.quad = np.array([[0.44594849091597, 0.4459849091597 , 0.22338158967801],
        [0.44594849091597, 0.10810301816807, 0.22338158967801],
        [0.10810301816807 ,0.44594849091597, 0.22338158967801],
        [0.09157621350977 ,0.09157621350977, 0.10995174365532],
        [0.09157621350977 ,0.81684757298046, 0.10995174365532],
        [0.81684757298046 ,0.09157621350977, 0.10995174365532]])

        self.domain_mark,self.bdry_mark = self.identify_bdry()
        self.clean_nodes = self.clean_node_identify()
        #shape: (6,3). 6 quadrature points, 2 coordinates + 1 weight.

    def clean_node_identify(self):
        flags = np.zeros([len(self.crd),1])
        for i in range(self.nrElem):
            flags[self.em[i,:]] = 1

        return np.where(flags>0.5)[0]
    

    def identify_bdry(self):
        ''' 
        boundary marker for L-shaped domain

        6 edges in total:

        1: x:[0,1], y=0     --> length=1
        2: x=1,     y:[0,1] --> length=1
        3: x:[-1,1],y=1     --> length=2
        4: x=-1,    y:[-1,1]--> length=2
        5: x:[-1,0],y=-1    --> length=1
        6: x=0,     y:[-1,0]--> length=1

        '''
        domain_mark = list()
        bdry_mark = list()
        clean_nodes = self.clean_node_identify()
        for ind in range(len(self.crd)):
            if ind in clean_nodes:
                if not (
                    (near(self.crd[ind,1],0.0) and self.crd[ind,0]>=0.)
                    or (near(self.crd[ind,0],1.0) and self.crd[ind,1]>=0.)
                    or near(self.crd[ind,1],1.0) 
                    or near(self.crd[ind,0],-1.0)
                    or (near(self.crd[ind,1],-1.0) and self.crd[ind,0]<=0.)
                    or (near(self.crd[ind,0],0.0) and self.crd[ind,1]<=0.)
                    ):
                    domain_mark.append(ind)
                else:
                    bdry_mark.append(ind)
        return domain_mark,bdry_mark

    def get_coef_mat(self,cd):
        #coord = [[x1,y1],[x2,y2],[x3,y3]], shape:(3,2)
        aug_cd = np.concatenate([np.ones((3,1)),cd],axis=1)
        L1 = np.linalg.solve(aug_cd,[1,0,0])
        L2 = np.linalg.solve(aug_cd,[0,1,0])
        L3 = np.linalg.solve(aug_cd,[0,0,1])
        
        return np.array([L1,L2,L3])
    
    
    def area(self,cx,cy):
        area = 0.5*np.abs(cx[0]*(cy[1]-cy[2])+cx[1]*(cy[2]-cy[0])+cx[2]*(cy[0]-cy[1])) #area is correct
        return area

    def quadrature(self,cx,cy,f,L,area=None):
        if area is None:
            area = 0.5*np.abs(cx[0]*(cy[1]-cy[2])+cx[1]*(cy[2]-cy[0])+cx[2]*(cy[0]-cy[1]))
        xq = (cx[0]*(1-self.quad[:,0]-self.quad[:,1]) + cx[1]*self.quad[:,0] + cx[2]*self.quad[:,1])
        yq = (cy[0]*(1-self.quad[:,0]-self.quad[:,1]) + cy[1]*self.quad[:,0] + cy[2]*self.quad[:,1])
        return area*np.sum(self.quad[:,2]*(f(xq,yq)*(L[0]+L[1]*xq+L[2]*yq)))
    
    def quadrature_points(self,cx,cy):
        xq = (cx[0]*(1-self.quad[:,0]-self.quad[:,1]) + cx[1]*self.quad[:,0] + cx[2]*self.quad[:,1])
        yq = (cy[0]*(1-self.quad[:,0]-self.quad[:,1]) + cy[1]*self.quad[:,0] + cy[2]*self.quad[:,1])
        return xq,yq
    
    def assemble(self,element_wise_staff):
    #shape of input: (nrElem,3,1)
    #assembling takes much time!!!
        node_wise_staff = torch.zeros([self.nrNode])
        node_wise_staff.index_add_(0,self.em_flattened,element_wise_staff.reshape(-1))

        return node_wise_staff.unsqueeze(-1)#[domain_mark]

    def get_full_quad(self):
        '''
        Returns dictionary:
            {
                'quadrature_points':[[x_i,y_i]]. numpy array with shape (nrElem,6,2) 
                'weight': [w_i]. numpy array with shape (nrElem,6). Area will be multiplied here.
                'coef_mat': [[a_i,b_i,c_i]]. numpy array with shape (nrElem,3,3). Each element has a 3x3 coef mat.
    
            }
        
        '''
        qp = []
        w = [] 
        cm = []
        for ind in np.arange(0,self.nrElem):
            curElem = self.em[ind,:]
            curCoord = self.crd[curElem,:] #shape: (3,2)

            coef_mat = self.get_coef_mat(curCoord) #[L1,L2,L3]

            area = self.area(curCoord[:,0],curCoord[:,1])
            x = (curCoord[0,0]*(1-self.quad[:,0]-self.quad[:,1]) + curCoord[1,0]*self.quad[:,0] + curCoord[2,0]*self.quad[:,1])
            y = (curCoord[0,1]*(1-self.quad[:,0]-self.quad[:,1]) + curCoord[1,1]*self.quad[:,0] + curCoord[2,1]*self.quad[:,1])
            
            emw = self.quad[:,2] * area

            qp.append(np.array([x,y]).T) #check shape
            w.append(emw) 
            cm.append(coef_mat) #Coef mat is correct (checked)

        output = {
            "quadrature_points":np.array(qp),
            "weight":np.array(w),
            "coef_mat":np.array(cm)
        }
        return output
    

class recorder(object):
    def __init__(self,path):
        if not os.path.exists(path):
            with io.open(path,'w') as f:
                f.write(json.dumps({}))
                #print("new file written")

        with open (path) as f:
            self.rec = json.load(f)
            print(self.rec)
            
        self.path = path
    
    def write(self,dict):
        self.rec.update(dict)
        #self.rec.append(dict)

    def save(self):
        with open(self.path,'w') as f:
            json.dump(self.rec,f)
        return True