from __future__ import annotations
import torch, math
from HTorch.manifolds import Euclidean, PoincareBall, Lorentz, HalfSpace, Manifold, Sphere
from torch import Tensor
from torch.nn import Parameter
import functools
from typing import Union
from HTorch.MCTensor import MCTensor

manifold_maps = {
    'Euclidean': Euclidean, 
    'PoincareBall': PoincareBall,
    'Lorentz': Lorentz, 
    'HalfSpace': HalfSpace,
    'Sphere':Sphere
}
__all__ = [
    'MCHTensor',
]

class MCHTensor(MCTensor):
    @staticmethod
    def __new__(cls, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
        ret = super().__new__(cls, *args, **kwargs)
        ret.manifold: Manifold = manifold_maps[manifold]()
        ret.curvature = curvature
        return ret

    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold: Manifold = manifold_maps[manifold]()
        self.curvature = curvature
    
    def clone(self, *args, **kwargs) -> MCHTensor:
        ## may be removed? to test
        new_obj = MCHTensor(super().clone(*args, **kwargs),
                          manifold=self.manifold.name, curvature=self.curvature)
        return new_obj

    def to(self, *args, **kwargs) -> MCHTensor:
        new_obj = MCHTensor([], manifold=self.manifold.name,
                          curvature=self.curvature)
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj
    
    def __repr__(self):
        return "{}, manifold={}, curvature={}".format(
            super().__repr__(), self.manifold.name, self.curvature)

    def to_other_manifold(self, name: str) -> MCHTensor:
        """Convert to the same point on the other manifold."""
        assert name != self.manifold.name
        if name == 'Lorentz':
            ret = self.manifold.to_lorentz(self, abs(self.curvature))
        elif name == 'HalfSpace':
            ret = self.manifold.to_halfspace(self, abs(self.curvature))
        elif name == 'PoincareBall':
            ret = self.manifold.to_poincare(self, abs(self.curvature))
        else:
            raise NotImplemented
        ret.manifold = manifold_maps[name]()
        return ret

    def Hdist(self, other: MCHTensor) -> Tensor:
        """Computes hyperbolic distance to other."""
        assert self.curvature == other.curvature, "Inputs should in models with same curvature!"
        if self.manifold.name == other.manifold.name:
            dist = self.manifold.distance(self, other, abs(self.curvature))
        else:
            #### transform to a self's manifold, combine with lazy evaulation?
            other_ = other.to_other_manifold(self.manifold.name)
            dist = self.manifold.distance(self, other_, abs(self.curvature))
        return dist.as_subclass(Tensor)

    def proj(self) -> MCHTensor:
        """Projects point p on the manifold."""
        return self.manifold.proj(self, abs(self.curvature))

    def proj_(self) -> MCHTensor:
        """Projects point p on the manifold."""
        return self.data.copy_(self.proj())

    def proj_tan(self, u: Tensor) -> Tensor:
        """Projects u on the tangent space of p."""
        return self.manifold.proj_tan(self, u, abs(self.curvature)).as_subclass(Tensor)

    def proj_tan0(self, u: Tensor) -> Tensor:
        """Projects u on the tangent space of the origin."""
        return self.manifold.proj_tan0(u, abs(self.curvature)).as_subclass(Tensor)

    def expmap(self, x: MCHTensor, u: Tensor) -> MCHTensor:
        """Exponential map."""
        return self.manifold.expmap(x, u, abs(self.curvature))

    def expmap0(self, u: Tensor) -> MCHTensor:
        """Exponential map, with x being the origin on the manifold."""
        res = self.manifold.expmap0(
            u, abs(self.curvature)).as_subclass(MCHTensor)
        res.manifold = self.manifold
        res.curvature = self.curvature
        return res

    def logmap(self, x: MCHTensor, y: MCHTensor) -> Tensor:
        """Logarithmic map, the inverse of exponential map."""
        return self.manifold.logmap(x, y, abs(self.curvature)).as_subclass(Tensor)

    def logmap0(self, y: MCHTensor) -> Tensor:
        """Logarithmic map, where x is the origin."""
        return self.manifold.logmap0(y, abs(self.curvature)).as_subclass(Tensor)

    def mobius_add(self, x: MCHTensor, y: MCHTensor, dim: int = -1) -> MCHTensor:
        """Performs hyperboic addition, adds points x and y."""
        return self.manifold.mobius_add(x, y, abs(self.curvature), dim=dim)

    def mobius_matvec(self, m: Tensor, x: MCHTensor) -> MCHTensor:
        """Performs hyperboic martrix-vector multiplication to m (matrix)."""
        return self.manifold.mobius_matvec(m, x, abs(self.curvature))

    def check_(self) -> Tensor:
        """Check if point on the specified manifold, project to the manifold if not."""
        check_result = self.manifold.check(
            self, abs(self.curvature)).as_subclass(Tensor)
        if not check_result:
            print('Warning: data not on the manifold, projecting ...')
            self.proj_()
        return check_result
    
    @staticmethod
    def find_mani_cur(args):
        for arg in args:
            if isinstance(arg, list):
                # Recursively apply the function to each element of the list
                manifold, curvature = MCHTensor.find_mani_cur(arg)
                break
            elif isinstance(arg, MCHTensor):
                manifold, curvature = arg.manifold, arg.curvature
                break
        return manifold, curvature
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        tmp = super().__torch_function__(func, types, args, kwargs)
        if type(tmp) in [MCTensor, MCHTensor] and not hasattr(tmp, 'manifold'):
            ret = cls(tmp)
            ret._nc, ret.res = tmp.nc, tmp.res
            ret.manifold, ret.curvature = cls.find_mani_cur(args)
            return ret
        return tmp