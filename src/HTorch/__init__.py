from HTorch import optimizers
from HTorch import layers
from HTorch import manifolds
from HTorch.MCTensor import MCTensor
from HTorch.HTensor import HTensor, HParameter


def get_manifold(manifold_name: str) -> manifolds.Manifold:
    """
    This function takes the manifold name as input and return the corresponding manifold
    """
    return {
        'Euclidean': manifolds.Euclidean, 
        'PoincareBall': manifolds.PoincareBall,
        'Lorentz': manifolds.Lorentz, 
        'HalfSpace': manifolds.HalfSpace,
        'Sphere': manifolds.Sphere
    }[manifold_name]()
