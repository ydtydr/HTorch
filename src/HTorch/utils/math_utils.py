"""Math utils functions."""

import torch
import numpy as np

def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)

def sq_norm(x, keepdim=True):
    return torch.norm(x, p=2, dim=-1, keepdim=keepdim) ** 2

def inner_product(x, y, keepdim=True):
    return (x*y).sum(dim=-1, keepdim=keepdim)

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        res = grad_output / (input ** 2 - 1) ** 0.5
        res[res.isnan()] = 0
        return res

    
def sample_boundary(n_Bs, d, cls):
    if cls =='RandomUniform' or d>2:
        pre_b = torch.randn(n_Bs, d)
        b = pre_b/torch.norm(pre_b,dim=-1,keepdim=True)
    elif cls == 'FixedUniform':
        theta = torch.arange(0,2 * np.pi, 2*np.pi/n_Bs)
        b = torch.stack([torch.cos(theta), torch.sin(theta)],1)
    elif cls == 'RandomDisk':
        theta = 2 * np.pi * torch.rand(n_Bs)
        b = torch.stack([torch.cos(theta), torch.sin(theta)],1)
    else:
        raise NotImplementedError
    return b

def sample_boundary4D(out_channel, in_channel, n_Bs, d):
    pre_b = torch.randn(out_channel, in_channel, n_Bs, d)
    b = pre_b/torch.norm(pre_b, dim=-1,keepdim=True)
    return b

def PoissonKernel(X, b):
#     X = X.view(X.size(0), 1, X.size(-1))
    X = X.unsqueeze(-2)
    return (1 - torch.norm(X, 2, dim=-1)**2)/(torch.norm(X-b, 2, dim=-1)**2)
#     return (1 - torch.sum(X * X, dim=-1))/torch.sum((X-b)**2,dim=-1)