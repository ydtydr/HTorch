"""
.. module:: manifolds
.. autoclass:: PoincareBall

"""
import torch
from HTorch.manifolds.base import Manifold
from HTorch.utils import artanh, tanh, arcosh, sq_norm, inner_product
import math
from torch import Tensor, device
from typing import Union

__all__ = ["PoincareBall"]

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.
    We use the following convention: x1^2 + ... + xd^2 < 1 / c
    Note that 1/sqrt(c) is the Poincare ball radius.
    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15} #embedding
        #self.eps = {torch.float32: 4e-3, torch.float64: 1e-5} #HGCN

    def origin(self, d:int, c:Union[float,Tensor], size:tuple=None, 
               device:device= None) -> Tensor:
        return torch.zeros(*size, d, device=device)

    def check(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        return torch.all(torch.norm(x, p=2, dim=-1) < 1 / math.sqrt(c))

    def sqdist(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        return self.distance(x, y, c) ** 2
    
    def distance(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqrt_c = c ** 0.5
        # sqnorm = sq_norm(x - y)
        # dist_c = arcosh(1 + 2 * c * sqnorm / ((1 - c * sq_norm(x)) * (1 - c * sq_norm(y))))
        # return dist_c / sqrt_c
        ### old approach using mobius operations
        dist_c = artanh(
            sqrt_c * self.mobius_add(-x, y, c, dim=-
                                     1).norm(dim=-1, p=2, keepdim=False)
        )
        return dist_c * 2 / sqrt_c
    
    def _lambda_x(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Conformal factor at Poincare Ball point x, lamda_x, such that
        the Metric Tensor g(x) = lambda_x^2*I."""
        x_sqnorm = sq_norm(x.data)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def metric(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        return self._lambda_x(x, c) ** 2

    def egrad2rgrad(self, x:Tensor, dx:Tensor, c:Union[float,Tensor]) -> Tensor:
        d_x = dx.clone()
        x_ = x.clone()
        if d_x.is_sparse:
            d_x_values = d_x._values()
            x_values = x_.index_select(0, d_x._indices().squeeze())
        else:
            d_x_values = d_x
            x_values = x_
        d_x_values /= self.metric(x_values, c)
        return d_x 

    def proj(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Project a point outside manifold to the Poincare Ball manifold """
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return v

    def proj_tan0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        return v

    def expmap(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqrt_c = c ** 0.5
        v_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(x, c) * v_norm)
                * v
                / (sqrt_c * v_norm)
        )
        gamma_1 = self.mobius_add(x, second_term, c)
        return gamma_1

    def expmap0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(v.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * v / (sqrt_c * u_norm)
        return gamma_1

    def logmap(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        sub = self.mobius_add(-x, y, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(x, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def logmap0(self, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqrt_c = c ** 0.5
        p_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * y

    def mobius_add(self, x:Tensor, y:Tensor, c:Union[float,Tensor], 
                   dim:int=-1) -> Tensor:
        x2 = sq_norm(x)
        y2 = sq_norm(y)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m:Tensor, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.bool)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def mobius_scale(self, r:Union[float,Tensor], x:Tensor, 
                     c:Union[float,Tensor]) -> Tensor:
        """ Scales a Halfspace point x by r."""
        sqrt_c = math.sqrt(c)
        two_over_sqrt_c = 2 / sqrt_c
        x_norm = x.norm(dim=-1, keepdim=True, p=2)
        return two_over_sqrt_c * tanh(r * artanh(sqrt_c * x_norm)) * x / x_norm
        
    def init_weights(self, w:Tensor, c:Union[float,Tensor], 
                     irange:float=1e-5) -> Tensor:
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, x:Tensor, y:Tensor, v:Tensor, c:Union[float,Tensor], 
                  dim:int=-1) -> Tensor:
        """ TODO: (Don't know what this is) """
        x2 = sq_norm(x)
        y2 = sq_norm(y)
        xy = (x * y).sum(dim=dim, keepdim=True)
        xv = (x * v).sum(dim=dim, keepdim=True)
        yv = (y * v).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * xv * y2 + c * yv + 2 * c2 * xy * yv
        b = -c2 * yv * x2 - c * xv
        d = 1 + 2 * c * xy + c2 * x2 * y2
        return v + 2 * (a * x + b * y) / d.clamp_min(self.min_norm)

    def inner(self, u:Tensor, v:Tensor=None, x:Tensor=None, 
              c:Union[float,Tensor]=None, keepdim:bool=False) -> Tensor:
        if v is None:
            v = u
        return self.metric(x, c) * (u * v).sum(dim=-1, keepdim=keepdim)

    def norm_t(self, u:Tensor, x:Tensor=None, c:Union[float,Tensor]=None, 
               keepdim:bool=True) -> Tensor:
        return torch.sqrt(self.inner(u, u, x, c, keepdim=keepdim))

    def ptransp(self, x:Tensor, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, v, c) * lambda_x / lambda_y

    def ptransp0(self, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        lambda_y = self._lambda_x(y, c)
        return 2 * v / lambda_y.clamp_min(self.min_norm)
    
    def to_lorentz(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a Poincare Ball point to the same point on Lorentz."""
        sqrt_c = c**0.5
        x2 = sq_norm(x, keepdim=False)
        y_i = 2*x/(1-c*(x2.unsqueeze(-1))) 
        y_n_plus_1 = (1+c*x2)/(sqrt_c*(1-c*x2))
        y = torch.cat((y_i, y_n_plus_1.unsqueeze(-1)), dim=-1)
        return y
    
    def to_lorentz_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a Poincare Ball tangent vector to the same tangent vector on Lorentz."""
        lambda_x = self._lambda_x(x, c)
        sq_lambda_x = lambda_x.square()
        inner_x_v = inner_product(x,v)
        other_comps_1 = lambda_x * v
        other_comps_2 = c * sq_lambda_x * inner_x_v * x
        other_comps = other_comps_1 + other_comps_2
        tail_comp = c**0.5 * sq_lambda_x * inner_x_v
        return torch.cat((other_comps, tail_comp), dim=-1)
    
    def to_halfspace(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a Poincare Ball point to the same point on HalfSpace."""
        xn = x[..., -1]
        x2 = sq_norm(x, keepdim=False)
        sqrt_c = c**0.5
        bottom = (1 - 2*sqrt_c*xn + c*x2)
        y_i = 2*x[..., :-1]/bottom.unsqueeze(-1) 
        y_n = (1 - c*x2)/(sqrt_c*bottom)
        y = torch.cat((y_i, y_n.unsqueeze(-1)), dim=-1)
        return y
    
    def to_halfspace_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts a Poincare Ball tangent vector to the same tangent vector on HalfSpace."""
        sqrt_c = c**0.5
        vi, vn = v[..., :-1], v[..., -1].unsqueeze(-1)
        xi, xn = x[..., :-1], x[..., -1].unsqueeze(-1)
        mu_x = 2. / (1 - 2*sqrt_c*xn + c*sq_norm(x))
        sq_mu_x = mu_x.square()
        # ui when 1<=i<=n-1
        other_comps_1 = mu_x*vi
        other_comps_2 = sq_mu_x*c*inner_product(x, v, keepdim=True)*xi
        other_comps_3 = sqrt_c*sq_mu_x*vn*xi
        other_comps = other_comps_1 - other_comps_2 + other_comps_3
        # un
        diff_xnv_vnx = xn*v - vn*x
        inner_tmp = inner_product(x, sqrt_c * diff_xnv_vnx-v)
        tail_comp_1 = sqrt_c * sq_mu_x * inner_tmp
        tail_comp_2 = mu_x * vn * (1 + sqrt_c*mu_x*xn)
        tail_comp = tail_comp_1 + tail_comp_2
        return torch.cat((other_comps, tail_comp), dim=-1)
        
    def angle(self, u:Tensor, v:Tensor, c:Union[float,Tensor])-> Tensor:
        """Computes the angle between two tangent vectors."""
        return torch.arccos((u * v).sum(dim=-1) / (torch.norm(u, dim=-1) * torch.norm(v, dim=-1)))
    
    def angle_at_x(self, x:Tensor, y:Tensor, c:Union[float,Tensor])-> Tensor:
        """Computes the angle of y at x, i.e., angle Oxy"""
        norm_x = x.norm(2, dim=-1)
        norm_y = y.norm(2, dim=-1)
        dot_prod = (x * y).sum(dim=-1)
        edist = (x - y).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + c * norm_x ** 2) - norm_x ** 2 * (1 + c * norm_y ** 2))
        denom = (norm_x * edist * (1 + c**2 * norm_x**2 * norm_y**2 - 2 * c * dot_prod).sqrt())
        return (num / denom).clamp_(min=-1.0, max=1.0).acos().as_subclass(torch.Tensor)
#         return (num / denom).clamp_(min=-1 + self.eps[x.dtype], max=1 - self.eps[x.dtype]).acos().as_subclass(torch.Tensor)