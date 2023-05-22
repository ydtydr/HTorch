"""
.. module:: manifolds
.. autoclass:: Sphere

"""
import torch
from HTorch.manifolds.base import Manifold
from HTorch.utils import artanh, tanh, arcosh, sq_norm
import math
from torch import Tensor, device
from typing import Union

__all__ = ["Sphere"]

class Sphere(Manifold):
    """
    Sphere Manifold class.
    We use the following convention: x1^2 + ... + xd^2 = 1/c
    Note that 1/sqrt(c) is the sphere radius, c is the curvature.
    """

    def __init__(self, ):
        super(Sphere, self).__init__()
        self.name = 'Sphere'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 1e-8, torch.float64: 1e-12}
    
    def inner(self, u:Tensor, v:Tensor=None, x:Tensor=None,
              c:Union[float,Tensor]=None, keepdim:bool=False) -> Tensor:
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def norm_t(self, u:Tensor, x:Tensor=None, c:Union[float,Tensor]=None, 
               keepdim:bool=True) -> Tensor:
        return torch.sqrt(self.inner(u, u, x, c, keepdim=keepdim))
    
    def check(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        return torch.isclose(self.inner(x,x), torch.tensor(1./c))

    def sqdist(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return self.distance(x, y, c) ** 2

    def distance(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        r = 1. / math.sqrt(c)
        inner = self.inner(x, y) * c
        inner = inner.clamp(-1 + self.eps[x.dtype], 1 - self.eps[x.dtype])
        return r * torch.acos(inner)
    
    def proj(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        r = 1. / math.sqrt(c)
        return r * x / x.norm(dim=-1, keepdim=True)
    
    def proj_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        v = v - (x * v).sum(dim=-1, keepdim=True) * x * c
        return v
    
    def egrad2rgrad(self, x:Tensor, dx:Tensor, c:Union[float,Tensor]) -> Tensor:
        d_x = dx.clone()
        if d_x.is_sparse:
            d_x_values = d_x._values()
            x_values = x.index_select(0, d_x._indices().squeeze())
        else:
            d_x_values = d_x
            x_values = x
        d_x_values.data.copy_(self.proj_tan(x_values, d_x_values, c))
        return d_x 
    
    def expmap(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqrt_c = math.sqrt(c)
        norm_v = v.norm(dim=-1)
        exp = x * torch.cos(norm_v * sqrt_c) + (1. / sqrt_c) * v * torch.sin(norm_v * sqrt_c) / norm_v
        return exp
#         retr = self.proj(x + v, c)
#         cond = norm_v > self.eps[norm_v.dtype]
#         return torch.where(cond, exp, retr)
    
    def retr(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return self.proj(x + v, c)

    def ptransp(self, x:Tensor, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor: # this may need to look
        norm_v = v.norm(dim=-1)
        u = self.proj_tan(y, v, c)
        return norm_v * u / u.norm(dim=-1)
    
    def logmap(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        u = self.proj_tan(x, y - x, c)
        dist = self.distance(x, y, c)
        return u * dist / u.norm(dim=-1).clamp_min(self.eps[x.dtype])
#         cond = dist.gt(self.eps[x.dtype])
#         result = torch.where(
#             cond, u * dist / u.norm(dim=-1).clamp_min(self.eps[x.dtype]), u
#         )
#         return result
        
    def init_weights(self, w:Tensor, c:Union[float,Tensor]) -> Tensor:
        w.data.copy_(self.random_uniform(w.size(), c=c, dtype=w.dtype))
        return w
    
    def random_uniform(self, *size:tuple, c:Union[float,Tensor]=1.0, 
                       dtype:torch.dtype=None, device:device=None):
        tens = torch.randn(*size, device=device, dtype=dtype)
        return self.proj(tens, c)