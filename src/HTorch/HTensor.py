from __future__ import annotations
import torch, math
from HTorch.manifolds import Euclidean, PoincareBall, Lorentz, HalfSpace, Manifold, Sphere
from torch import Tensor
from torch.nn import Parameter


manifold_maps = {
    'Euclidean': Euclidean, 
    'PoincareBall': PoincareBall,
    'Lorentz': Lorentz, 
    'HalfSpace': HalfSpace,
    'Sphere':Sphere
}
__all__ = [
    'HTensor',
    'HParameter'
]

class HTensor(Tensor):
    @staticmethod
    def __new__(cls, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
        ret = super().__new__(cls, *args, **kwargs)
        ret.manifold: Manifold = manifold_maps[manifold]()
        ret.curvature = curvature
        return ret

    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, device=None):
        self.manifold: Manifold = manifold_maps[manifold]()
        self.curvature = curvature

    def __repr__(self):
        return "Hyperbolic {}, manifold={}, curvature={}".format(
            super().__repr__(), self.manifold.name, self.curvature)
    
    def to_other_manifold(self, name: str) -> HTensor:
        """Convert to the same point on the other manifold."""
        assert name != self.manifold.name
        if name == 'Lorentz':
            tmp = self.manifold.to_lorentz(self, abs(self.curvature))
        elif name == 'HalfSpace':
            tmp = self.manifold.to_halfspace(self, abs(self.curvature))
        elif name == 'PoincareBall':
            tmp = self.manifold.to_poincare(self, abs(self.curvature))
        else:
            raise NotImplemented
        tmp.manifold = manifold_maps[name]()
        return tmp
            
    def Hdist(self, other: HTensor) -> Tensor:
        """Computes hyperbolic distance to other."""
        assert self.curvature == other.curvature, "Inputs should in models with same curvature!"
        if self.manifold.name == other.manifold.name:
            dist = self.manifold.distance(self, other, abs(self.curvature))
        else:
            #### transform to a self's manifold, combine with lazy evaulation?
            other_ = other.to_other_manifold(self.manifold.name)
            dist = self.manifold.distance(self, other_, abs(self.curvature))
        return dist.as_subclass(Tensor)
    
    def norm(self, p:int=2, dim:int=-1, keepdim=False) -> Tensor:
        """Returns p-norm as Tensor type"""
        return torch.norm(self, dim=dim, p=p, keepdim=keepdim).as_subclass(Tensor)
        
    def proj(self) -> HTensor:
        """Projects point p on the manifold."""
        return self.manifold.proj(self, abs(self.curvature))
    
    def proj_(self) -> HTensor:
        """Projects point p on the manifold."""
        return self.data.copy_(self.proj())
    
    def proj_tan(self, u:Tensor) -> Tensor:
        """Projects u on the tangent space of p."""
        return self.manifold.proj_tan(self, u, abs(self.curvature)).as_subclass(Tensor)

    def proj_tan0(self, u:Tensor) -> Tensor:
        """Projects u on the tangent space of the origin."""
        return self.manifold.proj_tan0(u, abs(self.curvature)).as_subclass(Tensor)
    
    def expmap(self, x:HTensor, u:Tensor) -> HTensor:
        """Exponential map."""
        return self.manifold.expmap(x, u, abs(self.curvature))

    def expmap0(self, u:Tensor) -> HTensor:
        """Exponential map, with x being the origin on the manifold."""
        res = self.manifold.expmap0(u, abs(self.curvature)).as_subclass(HTensor)
        res.manifold = self.manifold 
        res.curvature = self.curvature
        return res
    
    def logmap(self, x:HTensor, y:HTensor) -> Tensor:
        """Logarithmic map, the inverse of exponential map."""
        return self.manifold.logmap(x, y, abs(self.curvature)).as_subclass(Tensor)

    def logmap0(self, y:HTensor) -> Tensor:
        """Logarithmic map, where x is the origin."""
        return self.manifold.logmap0(y, abs(self.curvature)).as_subclass(Tensor)
    
    def mobius_add(self, x:HTensor, y:HTensor, dim:int=-1) -> HTensor:
        """Performs hyperboic addition, adds points x and y."""
        return self.manifold.mobius_add(x, y, abs(self.curvature), dim=dim)

    def mobius_matvec(self, m:Tensor, x:HTensor) -> HTensor:
        """Performs hyperboic martrix-vector multiplication to m (matrix)."""
        return self.manifold.mobius_matvec(m, x, abs(self.curvature))
    
    def clone(self, *args, **kwargs) -> HTensor:
        new_obj = HTensor(super().clone(*args, **kwargs), manifold=self.manifold.name, curvature=self.curvature)
        return new_obj
    
    def check_(self) -> Tensor:
        """Check if point on the specified manifold, project to the manifold if not."""
        check_result = self.manifold.check(self, abs(self.curvature)).as_subclass(Tensor)
        if not check_result:
            print('Warning: data not on the manifold, projecting ...')
            self.proj_()
        return check_result
    
    @staticmethod
    def find_mani_cur(args):
        for arg in args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                # Recursively apply the function to each element of the list
                manifold, curvature = HTensor.find_mani_cur(arg)
                break
            elif isinstance(arg, HTensor):
                manifold, curvature = arg.manifold, arg.curvature
                break
        return manifold, curvature
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, HTensor) and not hasattr(ret, 'manifold'):
            ret.manifold, ret.curvature = cls.find_mani_cur(args)
        return ret

class HParameter(HTensor, Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, manifold='PoincareBall', curvature=-1.0, requires_grad=True):
        res = HTensor._make_subclass(cls, data, requires_grad)
        return res
    
    def __init__(self, x, manifold='PoincareBall', curvature=-1.0, device=None):
        if isinstance(x, HTensor):
            self.manifold = x.manifold
            self.curvature = x.curvature
        else:
            self.manifold = manifold_maps[manifold]()
            self.curvature = curvature
            
    def init_weights(self, irange=1e-5):
        # this irange need to be controled for different floating-point precision
        self.data.copy_(self.manifold.init_weights(self, abs(self.curvature), irange))