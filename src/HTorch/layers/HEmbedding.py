import torch
import torch.nn.functional as F
from HTorch.HTensor import HParameter, HTensor

# class HEmbedding(torch.nn.Embedding):
#     def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
#         super(HEmbedding, self).__init__(*args, **kwargs)
#         self.weight = HParameter(self.weight, manifold=manifold, curvature=curvature)
#         self.weight.init_weights()
    
#     def forward(self, input):
#         output = super().forward(input).as_subclass(HTensor)
#         output.manifold = self.weight.manifold
#         output.curvature = self.weight.curvature
#         return output
    
class HEmbedding(torch.nn.Embedding):
    def __init__(self, *args, manifold='PoincareBall', curvature=-1.0, **kwargs):
        super(HEmbedding, self).__init__(*args, **kwargs)
        self.weight = HParameter(self.weight, manifold=manifold, curvature=curvature)
        self.manifold = self.weight.manifold
        self.curvature = self.weight.curvature
        self.weight.init_weights()
    
    def forward(self, input):
        output = super().forward(input).as_subclass(HTensor)
        output.manifold = self.manifold
        output.curvature = self.curvature
        return output