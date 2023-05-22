"""
.. module:: manifolds
.. autoclass:: HalfSpace

"""
from HTorch.utils import artanh, tanh, cosh, sinh, arcosh, sq_norm, inner_product
import torch
from torch import Tensor, device
from HTorch.manifolds.base import Manifold
import math
from typing import Union

__all__ = ["HalfSpace"]

class HalfSpace(Manifold):
    """
    HalfSpace Manifold class.
    We use the following convention: (x1, ..., xd), xd>0
    where c is the negation of the curvature (c > 0).
    """
    def __init__(self, ):
        super(HalfSpace, self).__init__()
        self.name = 'HalfSpace'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}

    def origin(self, d:int, c:Union[float,Tensor], size:tuple=None, 
               device:device= None) -> Tensor:
        ret = torch.zeros(*size, d, device=device)
        ret[..., -1] = 1 / math.sqrt(c)
        return ret

    def check(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Check the validity of the HalfSpace point sets."""
        return torch.all(x[..., -1] > 0) 

    def sqdist(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return torch.clamp(self.distance(x, y, c) ** 2, max=50.0)

    def distance(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqnorm = sq_norm(x-y)
        dist_c = arcosh(1 + sqnorm/(2*x[..., -1]*y[..., -1]).unsqueeze(-1))
        dist = dist_c / c**0.5
        return dist

    def _lambda_x(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Conformal factor at HalfSpace point x, lamda_x, such that
        the Metric Tensor g(x) = lambda_x^2*I."""
        sqrt_c = c**0.5
        xn = x[..., -1:]
        return 1/(sqrt_c*xn) 

    def metric(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        return self._lambda_x(x, c)**2
    
    def egrad2rgrad(self, x:Tensor, dx:Tensor, c:Union[float,Tensor]) -> Tensor:
        d_x = dx.clone()
        x_ = x.clone()
        if d_x.is_sparse:
            d_x_values = d_x._values()
            x_values = x_.index_select(0, d_x._indices().squeeze())
        else:
            d_x_values = d_x
            x_values = x_
        sqpn = (x_values[..., -1]**2).unsqueeze(-1)
        ### this is for a fake rgrad because the multiplication with c*pn instead of c*pn^2 for a clean exp map
        #sqpn = p[..., -1].unsqueeze(-1)
        d_x_values.mul_(c * sqpn) ### transform from Euclidean grad to Riemannian grad
        return d_x

    def proj(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Project a point outside manifold to the manifold, currently not supported 
        by HalfSpace, raise RuntimeError if point not on HalfSpace """
        ## raise a warning if xn<=0
#         assert torch.all(x[..., -1] > 0), "Point not on manifold, x[..., -1]<=0"
        y = x.clone()
        y.data[..., -1].clamp_(min=self.eps[x.dtype])
        return y

    def proj_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        ### HalfSpace's tangent space is R^n, proj_tan will not change v
        return v

    def proj_tan0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return v

    def expmap(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        x_n = x[..., -1].unsqueeze(-1)
        v = v / x_n
        v_n = v[..., -1].unsqueeze(-1)
        ### old way, not numerical robust now
        v_norm = torch.norm(v, p=2, dim=-1)
        v_norm_equals_zero = (v_norm == 0)
        v_norm = v_norm.clamp(min=1e-10)
        # # 1..n-1 components
        bottom_other_comps = torch.div(v_norm, tanh(v_norm)).unsqueeze(-1) - v_n
        other_comps = x[..., :-1] + torch.div(v[..., :-1], bottom_other_comps) * x_n
        # # last component
        tmp_tail_comp = torch.div(sinh(v_norm), v_norm).unsqueeze(-1) * v_n
        tail_comp = x_n / (cosh(v_norm).unsqueeze(-1) - tmp_tail_comp)
        y = torch.cat((other_comps, tail_comp), dim=-1)
        y[v_norm_equals_zero] = x[v_norm_equals_zero]
        return y

    def expmap0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqrt_c = (c**0.5)
        v = sqrt_c * v
        v_n = v[..., -1].unsqueeze(-1)
        ### may not numerical robust now
        v_norm = torch.norm(v, p=2, dim=-1)
        v_norm_equals_zero = (v_norm == 0)
        v_norm = v_norm.clamp(min=1e-10)
        # # 1..n-1 components
        bottom_other_comps = torch.div(v_norm, tanh(v_norm)).unsqueeze(-1) - v_n
        other_comps =  torch.div(v[..., :-1], bottom_other_comps)/sqrt_c
        # # last component
        tmp_tail_comp = torch.div(sinh(v_norm), v_norm).unsqueeze(-1) * v_n
        tail_comp = 1 / ((cosh(v_norm).unsqueeze(-1) - tmp_tail_comp) * sqrt_c)
        y = torch.cat((other_comps, tail_comp), dim=-1)
        y[v_norm_equals_zero] = self.origin(v.shape[-1], c, (1,), device=y.device)
        return y

    def logmap(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        v_norm = c**0.5 * self.distance(x, y, c).squeeze(-1)
        v_norm_equals_zero = (v_norm == 0)
        v_norm = v_norm.clamp(min=1e-10)
        # # 1..n-1 components
        other_comps_tmp1 = torch.div(v_norm, sinh(v_norm)) * torch.div(x[..., -1], y[..., -1])
        other_comps =  other_comps_tmp1.unsqueeze(-1) * (y[..., :-1] - x[..., :-1])
        # # last component
        tail_comp = torch.div(v_norm, tanh(v_norm)) - torch.div(v_norm, sinh(v_norm)) * torch.div(x[..., -1], y[..., -1])
        tail_comp = tail_comp * x[..., -1]
        v = torch.cat((other_comps, tail_comp.unsqueeze(-1)), dim=-1)
        v[v_norm_equals_zero] = 0
        return v 
    
    def logmap0(self, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        origin = torch.zeros_like(y)
        origin[..., -1] = 1 / math.sqrt(c)
        return self.logmap(origin, y, c)

    def inner(self, u:Tensor, v:Tensor=None, x:Tensor=None, 
              c:Union[float,Tensor]=None, keepdim:bool=True) -> Tensor:
        if v is None:
            v = u
        metric = self.metric(x, c)
        return metric * (u * v).sum(dim=-1, keepdim=keepdim)

    def norm_t(self, u:Tensor, x:Tensor=None, c:Union[float,Tensor]=None, 
               keepdim:bool=True) -> Tensor:
        return torch.sqrt(self.inner(u, u, x, c, keepdim=keepdim))
    
    def mobius_add(self, x:Tensor, y:Tensor, c:Union[float,Tensor], dim:int=-1) -> Tensor:
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(x, v, c)

    def mobius_matvec(self, m:Tensor, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)
    
    def init_weights(self, w:Tensor, c:Union[float,Tensor], irange:float=1e-5) -> Tensor:
        w.data.uniform_(-irange, irange)
        w[...,-1].data.mul_(1/(c**0.5))
        w[...,-1].data.add_(1/(c**0.5))
        return w

    def ptransp(self, x:Tensor, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.inner(logxy, v=v, x=x, c=c) / sqdist
        res =  v - alpha * (logxy + logyx)
        return self.proj_tan(y, res, c)

    def ptransp0(self, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        origin = torch.zeros_like(y)
        origin[..., -1] = 1 / math.sqrt(c)
        return self.ptransp(origin, y, v, c)

    def to_lorentz(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a HalfSpace point to the same point on Lorentz."""
        sqrt_c = c**0.5
        xn = x[..., -1]
        x2 = (x[...,:-1].square()).sum(dim=-1)
        y_i = x[..., :-1]/(sqrt_c*(xn.unsqueeze(-1)))
        y_n = (xn + x2/xn - 1/(c*xn))/2
        y_n_plus_1 = (xn + x2/xn + 1/(c*xn))/2
        y = torch.cat((y_i, y_n.unsqueeze(-1), y_n_plus_1.unsqueeze(-1)), dim=-1)
        return y

    def to_lorentz_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a HalfSpace tangent vector to the same tangent vector on Lorentz."""
        # sum of xj^2 where j=1 to n-1
        sqrt_c = c**0.5
        xi = x[..., :-1]
        vi = v[..., :-1]
        xn = x[..., -1].unsqueeze(-1)
        sq_xn = xn.square()
        sq_norm_xi = sq_norm(xi)
        vn = v[..., -1].unsqueeze(-1)
        other_comps =  (1/(sqrt_c*xn))*vi - vn*xi/(sqrt_c*sq_xn) 
        inv_cx2n = 1/(c*sq_xn)
        tail_comp_tmp1 = inner_product(xi, vi)/xn
        tail_comp_tmp2 =  1 - sq_norm_xi/sq_xn
        tail_comp_first = tail_comp_tmp1 + (tail_comp_tmp2 + inv_cx2n)*vn/2
        tail_comp_second = tail_comp_tmp1 + (tail_comp_tmp2 - inv_cx2n)*vn/2
        return  torch.cat((other_comps, tail_comp_first, tail_comp_second), dim=-1) 

    def to_poincare(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a HalfSpace point to the same point on Poincare Ball."""
        xn = x[..., -1]
        x2 = (x.square()).sum(dim=-1)
        sqrt_c = c**0.5
        bottom = (1 + 2*sqrt_c*xn + c*x2)
        y_i = 2*x[..., :-1]/bottom.unsqueeze(-1) 
        y_n = (c*x2 - 1)/(sqrt_c*bottom)
        y = torch.cat((y_i, y_n.unsqueeze(-1)), dim=-1)
        return y
    
    def to_poincare_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Convert a HalfSpace tangent vector to the same tangent vector on Poincare Ball."""
        sqrt_c = c**0.5
        vi, vn = v[..., :-1], v[..., -1].unsqueeze(-1)
        xi, xn = x[..., :-1], x[..., -1].unsqueeze(-1)
        mu_x = 2. / (1 + 2*sqrt_c*xn + c*sq_norm(x))
        sq_mu_x = mu_x.square()
        # ui when 1<=i<=n-1
        other_comps_1 = mu_x*vi
        other_comps_2 = sq_mu_x*c*inner_product(x, v, keepdim=True)*xi
        other_comps_3 = sqrt_c*sq_mu_x*vn*xi
        other_comps = other_comps_1 - other_comps_2 - other_comps_3
        # un
        diff_xnv_vnx = xn*v - vn*x
        inner_tmp = inner_product(x, v+sqrt_c*diff_xnv_vnx)
        tail_comp_1 = sqrt_c*sq_mu_x*inner_tmp
        tail_comp_2 = mu_x*vn*(1-sqrt_c*mu_x*xn)
        tail_comp = tail_comp_1 + tail_comp_2
        return torch.cat((other_comps, tail_comp), dim=-1)