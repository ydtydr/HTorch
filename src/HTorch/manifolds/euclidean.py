"""
.. module:: manifolds
.. autoclass:: Euclidean

"""
from HTorch.manifolds.base import Manifold
import torch
from torch import Tensor, device
from typing import Union

__all__ = ["Euclidean"]

class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """
    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def origin(self, d:int, c:Union[float,Tensor], size:tuple=None, 
               device:device= None) -> Tensor:
        return torch.zeros(*size, d, device=device)

    def normalize(self, x:Tensor, power:int=2, maxnorm:float=1.0) -> Tensor:
        """Normalize the input so that the power-norm along the last dimension is lower than maxnorm."""
        dim = x.size(-1)
        x.view(-1, dim).renorm_(power, 0, maxnorm)
        return x

    def norm_t(self, u:Tensor, x:Tensor=None, c:Union[float,Tensor]=None, 
               keepdim:bool=True) -> Tensor:
        return u.norm(p=2, keepdim=keepdim)

    def inner(self, u:Tensor, v:Tensor=None, x:Tensor=None,
              c:Union[float,Tensor]=None, keepdim:bool=False) -> Tensor:
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def metric(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        raise NotImplementedError

    def distance(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return (x - y).pow(2).sum(dim=-1).pow(1/2)
    
    def sqdist(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return self.distance(x, y, c) ** 2

    def egrad2rgrad(self, x:Tensor, dx:Tensor, c:Union[float,Tensor]) -> Tensor:
        return dx

    def proj(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        return x

    def proj_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return v

    def proj_tan0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return v

    def expmap(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return x + v

    def expmap0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return v
    
    def logmap(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return y - x

    def logmap0(self, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return y

    def mobius_add(self, x:Tensor, y:Tensor, c:Union[float,Tensor], 
                   dim:int=-1) -> Tensor:
        return x + y

    def mobius_matvec(self, m:Tensor, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w:Tensor, c:Union[float,Tensor], 
                     irange:float=1e-5) -> Tensor:
        w.data.uniform_(-irange, irange)
        return w

    def ptransp(self, x:Tensor, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return v

    def ptransp0(self, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return y + v