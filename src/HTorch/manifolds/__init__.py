"""
.. module:: manifolds

The `manifolds` module contains different hyperbolic manifolds.

"""

from HTorch.manifolds.euclidean import Euclidean
from HTorch.manifolds.poincare import PoincareBall
from HTorch.manifolds.lorentz import Lorentz
from HTorch.manifolds.halfspace import HalfSpace
from HTorch.manifolds.sphere import Sphere
from HTorch.manifolds.base import Manifold

__all__ = ['Euclidean', 'PoincareBall', 'Lorentz', 'HalfSpace', 'Sphere', 'Manifold']