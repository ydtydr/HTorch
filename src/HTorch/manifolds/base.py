"""
.. module:: manifolds
.. autoclass:: Manifold

"""
from torch import Tensor, device
from typing import Union

__all__=["Manifold"]

class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """
    def __init__(self):
        """Initialize"""
        super().__init__()
        self.eps=10e-8

    def origin(self, d:int, c:Union[float,Tensor], size:tuple=None, 
               device:device= None) -> Tensor:
        """The origin in the manifold"""
        raise NotImplementedError

    def distance(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Distance between pairs of points on the same manifold."""
        raise NotImplementedError
        
    def sqdist(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Squared distance between pairs of points on the same manifold."""
        raise NotImplementedError
    
    def metric(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Metric tensor on the tangent space of x."""
        raise NotImplementedError

    def egrad2rgrad(self, x:Tensor, dx:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Projects point x on the manifold."""
        raise NotImplementedError

    def proj_tan(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Project v to the tangent space of a point x on the manifold"""
        raise NotImplementedError

    def proj_tan0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Projects v on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, x:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Exponential map, takes a tangent vector v of a point x on the 
        manifold to a point y on the manifold."""
        raise NotImplementedError
    
    def expmap0(self, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Exponential map of v at the origin."""
        raise NotImplementedError
        
    def logmap(self, x:Tensor, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Logarithmic map, the inverse of exponential map. Logmap returns 
        the tangent vector v at a point x on the manifold so as to reach y on the 
        manifold, such that expmap(x, logmap(x,y,c), c)=y """
        raise NotImplementedError

    def logmap0(self, y:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Logarithmic map of point y at the origin."""
        raise NotImplementedError

    def mobius_add(self, x:Tensor, y:Tensor, c:Union[float,Tensor], 
                   dim:int=-1) -> Tensor:
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m:Tensor, x:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w:Tensor, c:Union[float,Tensor], 
                     irange:float=1e-5) -> Tensor:
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, u:Tensor, v:Tensor=None, x:Tensor=None, 
              c:Union[float,Tensor]=None, keepdim:bool=True) -> Tensor:
        """Inner product for tangent vectors at point x (if given)."""
        raise NotImplementedError

    def ptransp(self, x:Tensor, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Parallel transport, generalization of translation in the hyperbolic space,
        u = ptransp(x,y,v,c) maps v in tangent space of x to the tangent space of y 
        along the geodesic connecting x and y."""
        raise NotImplementedError

    def ptransp0(self, y:Tensor, v:Tensor, c:Union[float,Tensor]) -> Tensor:
        """Parallel transport of v from the origin to y."""
        raise NotImplementedError

    def norm_t(self, u:Tensor, x:Tensor=None, c:Union[float,Tensor]=None, 
               keepdim:bool=True) -> Tensor:
        """Norm of the tangent vector in tangent space."""
        raise NotImplementedError

