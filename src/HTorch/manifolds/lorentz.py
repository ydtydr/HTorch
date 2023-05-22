"""
.. module:: manifolds
.. autoclass:: Lorentz

"""
import torch
from HTorch.manifolds.base import Manifold
import math
from HTorch.utils import arcosh, cosh, sinh, sq_norm
from torch import Tensor, device
from typing import Union

__all__ = ["Lorentz"]

class Lorentz(Manifold):
    """
    Lorentz manifold class.

    We use the following convention: x1^2 + ... + xd^2 -x_{d+1}^2 = -1/c

    c is the hyperbolic curvature
    """

    def __init__(self, ):
        super(Lorentz, self).__init__()
        self.name = 'Lorentz'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e7

    def check(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        return torch.all(
            torch.isclose(self.inner(x,x), torch.tensor(-1./c, dtype=x.dtype)))
    
    def check_tan(self, x, x_tan, c):
        res = self.inner(x, x_tan, c=c)
        return torch.all(torch.isclose(res, torch.zeros_like(res)))
        
    def origin(self, d, c, size=None):
        ret = torch.zeros(*size, d + 1)
        ret[..., -1] = math.sqrt(1 / c)
        return ret

    def metric(self, x:Tensor, c:Union[float,Tensor], d:int) -> Tensor:
        ret = torch.eye(d)
        ret[-1, -1] = -1
        return ret

    def inner(self, u:Tensor, v:Tensor=None, x:Tensor=None, 
              c:Union[float,Tensor]=None, keepdim:bool=True) -> Tensor:
        if v is None:
            v = u    
        res = torch.sum(u[...,:-1] * v[...,:-1], dim=-1) -  u[...,-1] * v[...,-1]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def norm_t(self, u:Tensor, x:Tensor=None, c:Union[float,Tensor]=None, 
               keepdim:bool=True) -> Tensor:
        dot = self.inner(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return torch.clamp(self.distance(x, y, c) ** 2, max=50.0)
    
    def distance(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        prod = self.inner(x, y)
        theta = torch.clamp(-prod * c, min=1.0 + self.eps[x.dtype])
        dist = (1./c) ** 0.5 * arcosh(theta)
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return dist
    
    def egrad2rgrad(self, x:Tensor, dx:Tensor, c:Union[float,Tensor]) -> Tensor: # non_inplace
        d_x = dx.clone()
        x_ = x.clone()
        if d_x.is_sparse:
            d_x_values = d_x._values()
            x_values = x_.index_select(0, d_x._indices().squeeze())
        else:
            d_x_values = d_x
            x_values = x_
        d_x_values.narrow(-1, -1, 1).mul_(-1)
        # adopt x + c<x, dx>_L * x
        d_x_values.addcmul_(c * self.inner(x_values, d_x, keepdim=True).expand_as(x_values), x_values)
        return d_x
    
    def proj(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Project a point outside manifold to the Lorentz manifold """
        d = x.size(-1) - 1
        y = x.narrow(-1, 0, d)
        y_sqnorm = sq_norm(y)[..., 0] 
        mask = torch.ones_like(x)
        mask[..., -1] = 0
        vals = torch.zeros_like(x)
        vals[..., -1] = torch.sqrt(torch.clamp(1. / c + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x
    
    def proj_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        # not the standard way as x + c<x, dx>_L * x, here only modify the last dimension
        d = x.size(-1) - 1
        ux = torch.sum(x.narrow(-1, 0, d) * v.narrow(-1, 0, d), dim=-1)
        mask = torch.ones_like(v)
        mask[..., -1] = 0
        vals = torch.zeros_like(v)
        vals[..., -1] = ux / torch.clamp(x[..., -1], min=self.eps[x.dtype])
        return vals + mask * v

    def proj_tan0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        vals = torch.zeros_like(v)
        vals[..., -1] = v[..., -1]
        return v - vals

    def expmap(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        normu = self.norm_t(v)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = c ** 0.5 * normu
        theta = torch.clamp(theta, min=self.min_norm)
        y = cosh(theta) * x + sinh(theta) * v / theta
        return self.proj(y, c)

    def expmap0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        K = 1. / c
        sqrtK = K ** 0.5
        d = v.size(-1) - 1
        x = v.narrow(-1, 0, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        y = torch.ones_like(v)
        y[..., -1:] = sqrtK * cosh(theta)
        y[..., :-1] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(y, c)
    
    def logmap(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        xy = torch.clamp(self.inner(x, y) + 1./c, max=-self.eps[x.dtype]) - 1./c
        u = y + xy * x * c
        normu = self.norm_t(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        v = dist * u / normu
        return self.proj_tan(x, v, c)

    def logmap0(self, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        K = 1. / c
        sqrtK = K ** 0.5
        d = y.size(-1) - 1
        x = y.narrow(-1, 0, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        v = torch.zeros_like(y)
        theta = torch.clamp(y[..., -1:] / sqrtK, min=1.0 + self.eps[y.dtype])
        v[..., :-1] = sqrtK * arcosh(theta) * x / x_norm
        return v
        #TODO: why return self.proj_tan(x, v, c) will cause nan in cora_lp?

    def mobius_add(self, x:Tensor, y:Tensor, c:Union[float,Tensor], dim:int=-1) -> Tensor:
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(x, v, c)

    def mobius_matvec(self, m:Tensor, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x:Tensor, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.inner(logxy, v) / sqdist
        res = v - alpha * (logxy + logyx)
        return self.proj_tan(y, res, c)

    def ptransp0(self, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        K = 1. / c
        sqrtK = K ** 0.5
        y0 = y.narrow(-1, -1, 1)
        d = y.size(-1) - 1
        x = y.narrow(-1, 0, d)
        x_norm = torch.clamp(torch.norm(x, p=2, dim=-1), min=self.min_norm)
        x_normalized = x / x_norm.unsqueeze(-1)
        u = torch.ones_like(y)
        u[..., -1] = - x_norm 
        u[..., :-1] = (sqrtK - y0) * x_normalized
        alpha = torch.sum(x_normalized * v[..., :-1], dim=-1, keepdim=True) / sqrtK
        res = v - alpha * u
        return self.proj_tan(y, res, c)
    
    def init_weights(self, w:Tensor, c:Union[float,Tensor], irange:float=1e-5) -> Tensor:
        w.data.uniform_(-irange, irange)
        w.data.copy_(self.proj(w.data, c))
        return w

    def to_poincare(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a Lorentz point to the same point on Poincare Ball."""
        xn = x[..., -1]
        bottom = c ** 0.5 * xn.unsqueeze(-1) + 1
        y = x[..., :-1] / bottom
        return y
    
    def to_poincare_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Convert a Lorentz tangent vector to the same tangent vector on Poincare Ball."""
        sqrt_c = c ** 0.5
        x_n = x[..., -1].unsqueeze(-1)
        v_n = v[..., -1].unsqueeze(-1)
        tmp = sqrt_c*x_n + 1
        comp_1 = v[..., :-1]/tmp 
        comp_2 = ((sqrt_c*v_n)/tmp.square())*x[..., :-1]
        return comp_1-comp_2
    
    def to_halfspace(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a Lorentz point to the same point on HalfSpace."""
        sqrt_c = c ** 0.5
        diff = (x[..., -1] - x[..., -2]).unsqueeze(-1)
        denom = sqrt_c * diff
        x_i = x[..., :-2]
        y_i = x_i / denom
        y_n = 1 / (c * diff)
        y = torch.cat((y_i, y_n), dim=-1)
        return y

    def to_halfspace_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Convert a Lorentz tangent vector to the same tangent vector on HalfSpace."""
        sqrt_c = c ** 0.5
        x_i = x[..., :-2]
        v_i = v[..., :-2]
        # x_n - x_n-1
        x_diff = (x[..., -1] - x[..., -2]).unsqueeze(-1)
        sq_x_diff = x_diff.square()
        # v_n - v_n-1
        v_diff = (v[..., -2] - v[..., -1]).unsqueeze(-1)
        u = torch.zeros_like(x[..., :-1])
        # u_i, other comps
        u[..., :-1] = v_i/(sqrt_c*x_diff) + (v_diff/(sqrt_c*sq_x_diff))*x_i
        # u_n, tail comp
        u[..., -1] = (v_diff/(c*sq_x_diff)).squeeze(-1)
        return u