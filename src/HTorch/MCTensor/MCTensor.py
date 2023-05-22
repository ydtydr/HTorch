from __future__ import annotations
import torch
import math
from torch import Tensor
from torch.nn import Parameter
from HTorch.MCTensor.MCOpBasics import _Renormalize, _Simple_renormalize_old, _Grow_ExpN, _AddMCN,  _ScalingN,\
    _DivMCN, _MultMCN, _exp, _pow, _square, _sinh, _cosh, _tanh, _log, _exp, _sqrt, \
    _softmax, _log_softmax, _cross_entropy, _mse_loss, _layer_norm, _atanh, _log1p_standard, \
    _clamp, _norm, _sum, _mean
from HTorch.MCTensor.MCOpMatrix import _Dot_MCN, _MV_MC_T, _MV_T_MC, _MM_MC_T, _MM_T_MC,\
    _BMM_MC_T, _BMM_T_MC, _4DMM_T_MC, _4DMM_MC_T
from typing import Union, List
import functools

HANDLED_FUNCTIONS = {}

def implements(torch_function):
    """Register a torch function override for MCTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

class MCTensor(Tensor):
    @staticmethod 
    def __new__(cls, *args,  nc=1, **kwargs): 
        ret = super().__new__(cls, *args, **kwargs)
        ret._nc = nc
        ret.res = torch.zeros(ret.size() + (nc-1,), dtype=ret.dtype, device=ret.device)
        return ret

    def __init__(self, *args,  nc=1, **kwargs):
        self._nc = nc
        self.res = torch.zeros(self.size() + (nc-1,), dtype=self.dtype, device=self.device)
        
    @staticmethod
    def wrap_tensor_to_mctensor(tensor: Tensor) -> MCTensor:
        # involves copying
        ret = MCTensor(tensor[..., 0], nc=tensor.size(-1))
        ret.res.data.copy_(tensor[..., 1:].data)
        return ret

    @staticmethod
    def wrap_tensor_and_res_to_mctensor(val: Tensor, res: Tensor) -> MCTensor:
        # involves copying
        ret = MCTensor(val, nc=res.shape[-1] + 1)
        ret.res.data.copy_(res.data)
        return ret

    @property
    def tensor(self):
        return torch.cat([self.as_subclass(Tensor).view(*self.shape, 1), self.res], -1)
    
    @property
    def T(self):
        return torch.transpose(self, 0, 1)
    
    @property
    def nc(self):
        return self._nc
    
    def normalize_(self, simple=False):
        if simple:
            normalized_self = _Simple_renormalize_old(self.tensor, self.nc)
        else:
            normalized_self = _Renormalize(self.tensor, self.nc)
        self.as_subclass(Tensor).data.copy_(normalized_self[..., 0].data)
        self.res.data.copy_(normalized_self[..., 1:].data)
    
    """
    override python default methods
    """
    def __repr__(self, *args, **kwargs):
        return "{}, nc={}".format(super().__repr__(), self.nc)

    def __add__(self, other) -> MCTensor:
        ''' add self with other'''
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__add__(other)
        else:
            obj = torch.add(self, other)
        return obj

    def __radd__(self, other) -> MCTensor:
        ''' add self with other'''
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__radd__(other)
        else:
            obj = torch.add(self, other)
        return obj
    
    def __sub__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__sub__(other)
        else:
            obj = torch.add(self, -other)
        return obj

    def __rsub__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__rsub__(other)
        else:
            obj = torch.add(other, -self)
        return obj

    def __mul__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__mul__(other)
        else:
            obj = torch.mul(self, other)
        return obj

    def __rmul__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__rmul__(other)
        else:
            obj = torch.mul(other, self)
        return obj
    
    def __truediv__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__truediv__(other)
        else:
            obj = torch.div(self, other)
        return obj

    def __rtruediv__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__rtruediv__(other)
        else:
            obj = torch.div(other, self)
        return obj

    def __matmul__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__matmul__(other)
        else:
            obj = torch.matmul(self, other)
        return obj

    def __rmatmul__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__rmatmul__(other)
        else:
            obj = torch.matmul(other, self)
        return obj

    def __pow__(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            obj = super().__pow__(other)
        else:
            obj = torch.pow(self, other)
        return obj

    def __setitem__(self, key, value: Union[MCTensor, Tensor, int, float]):
        if isinstance(key, tuple) and Ellipsis in key:
            res_key = (*key, slice(None, None, None))
        else:
            res_key = key
        if isinstance(value, MCTensor):
            super().__setitem__(key, value.as_subclass(Tensor))
            self.res[res_key].data.copy_(value.res.data)
        else:
            super().__setitem__(key, value)
            self.res[res_key] = 0
    
    """
    override torch default functions
    """
    def dot(self, other) -> MCTensor:
        if self.nc == 1 and (not isinstance(other, MCTensor) or other.nc == 1):
            if isinstance(other, MCTensor):
                other = other.as_subclass(Tensor)
            obj = super().as_subclass(Tensor).dot(other)
        else:
            obj = torch.dot(self, other)
        return obj
    
    def abs(self) -> MCTensor:
        return torch.abs(self)
    
    def sum(self, dim=None, keepdim=False, **kw) -> MCTensor:
        return torch.sum(self, dim=dim, keepdim=keepdim)
    
    def mean(self, dim=None, keepdim=False, **kw) -> MCTensor:
        return torch.mean(self, dim=dim, keepdim=keepdim)

    def norm(self, dim=None, keepdim=False, p=2, **kw) -> MCTensor:
        return torch.norm(self, dim=dim, keepdim=keepdim, p=p)

    def sinh(self) -> MCTensor:
        return torch.sinh(self)
    
    def cosh(self) -> MCTensor:
        return torch.cosh(self)
    
    def tanh(self) -> MCTensor:
        return torch.tanh(self)

    def atanh(self) -> MCTensor:
        return torch.atanh(self)

    def clamp_min(self, min=None) -> MCTensor:
        return torch.clamp_min(self, min=min)

    def clamp_max(self, max=None) -> MCTensor:
        return torch.clamp_max(self, max=max)

    def clone(self) -> MCTensor:
        return torch.clone(self)

    def unsqueeze(self, *args, **kwargs) -> MCTensor:
        return torch.unsqueeze(self, *args, **kwargs)

    def squeeze(self, *args, **kwargs) -> MCTensor:
        return torch.squeeze(self, *args, **kwargs)

    def reshape(self, shape) -> MCTensor:
        if type(shape) == int:
            shape = (shape,)
        return torch.reshape(self, shape)

    def transpose(self, *args, **kwargs) -> MCTensor:
        return torch.transpose(self, *args, **kwargs)

    def narrow(self, dim, start, length) -> MCTensor:
        if dim < 0:
            dim = dim + self.dim()
        return super().narrow(dim, start, length)
    
    def index_select(self, dim, index) -> MCTensor:
        if dim < 0:
            dim = dim + self.dim()
        return super().index_select(dim, index)
    
    @staticmethod
    def replace_args(args):
        new_args = []
        for arg in args:
            if isinstance(arg, list):
                # Recursively apply the function to each element of the list
                new_arg = MCTensor.replace_args(arg)
            elif isinstance(arg, MCTensor):
                # Replace MCTensor with its `res` attribute
                new_arg = arg.res
            else:
                # For other types of objects, just use the original object
                new_arg = arg
            new_args.append(new_arg)
        return tuple(new_args)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # inherence nc
        if kwargs is None:
            kwargs = {}
        if func in HANDLED_FUNCTIONS: # torch.Tensor with MCTensor contents
            ret = HANDLED_FUNCTIONS[func](*args, **kwargs)
            if isinstance(ret, Tensor) and not isinstance(ret, MCTensor):
                return cls.wrap_tensor_to_mctensor(ret)
        else: # pytorch functions handle main and res component separately
            ret = super().__torch_function__(func, types, args, kwargs)
            if isinstance(ret, MCTensor) and not hasattr(ret, '_nc'):
                new_args = cls.replace_args(args)
                ret.res = super().__torch_function__(func, types, new_args, kwargs).as_subclass(Tensor)
                ret._nc = ret.res.size(-1) + 1
        return ret
    
@implements(torch.add)
def add(input: Union[MCTensor, int, float], other: Union[MCTensor, int, float], alpha=1) -> MCTensor:
    if type(input) == int or type(input) == float:
        input = torch.tensor(input, dtype=other.dtype,
                             device=other.device)
    if type(other) == int or type(other) == float:
        other = torch.tensor(other, device=input.device,
                             dtype=input.dtype)
    if alpha != 1:
        other = alpha * other  # should check for MCTensor multiplication
    if isinstance(input, MCTensor) and isinstance(other, MCTensor):
        return _AddMCN(input.tensor, other.tensor, simple=False)
    elif isinstance(input, MCTensor):
        x = input  # the MCTensor
        value = other
    else:
        x = other  # the MCTensor
        value = input
    return _Grow_ExpN(x.tensor, value)

@implements(torch.mul)
def mul(input: Union[MCTensor, int, float], other: Union[MCTensor, int, float]) -> MCTensor:
    if type(input) == int or type(input) == float:
        input = torch.tensor(input, dtype=other.dtype,
                             device=other.device)
    if type(other) == int or type(other) == float:
        other = torch.tensor(other, device=input.device,
                             dtype=input.dtype)
    normalize_case = 0 #case 0, not renormalize, explore carefully
    if isinstance(input, MCTensor) and isinstance(other, MCTensor):
        return _MultMCN(input.tensor, other.tensor, case=normalize_case)
    elif isinstance(input, MCTensor):
        x = input  # the MCTensor
        value = other
    else:
        x = other  # the MCTensor
        value = input
    return _ScalingN(x.tensor, value)

@implements(torch.div)
def div(x: Union[MCTensor, int, float], y: Union[MCTensor, int, float]) -> MCTensor:
    if type(x) == int or type(x) == float:
        x = torch.tensor(x, device=y.tensor.device, dtype=y.tensor.dtype)
    if type(y) == int or type(y) == float:
        y = torch.tensor(y, device=x.tensor.device, dtype=x.tensor.dtype)
    normalize_case = 2 #case 2, renormalize, explore carefully
    if isinstance(x, MCTensor) and type(y) == Tensor:
        nc = x.nc
        y_tensor = torch.zeros(y.size() + (nc,), device=y.device, dtype=y.dtype)
        y_tensor[..., 0] = y
        result = _DivMCN(x.tensor, y_tensor, case=normalize_case)
    elif type(x) == Tensor and isinstance(y, MCTensor):
        nc = y.nc
        x_tensor = torch.zeros(x.size() + (nc,), device=x.device, dtype=x.dtype)
        x_tensor[..., 0] = x
        result = _DivMCN(x_tensor, y.tensor, case=normalize_case)
    elif isinstance(x, MCTensor) and isinstance(y, MCTensor):
        result = _DivMCN(x.tensor, y.tensor, case=normalize_case)
    else:
        raise NotImplemented
    return result

@implements(torch.rand_like)
def rand_like(input: MCTensor, requires_grad=False, device=None, dtype=None) -> MCTensor:
    val = torch.randn_like(input.as_subclass(Tensor), requires_grad=requires_grad, device=device, dtype=dtype)
    return MCTensor(val, nc=input.nc)

@implements(torch.zeros_like)
def zeros_like(input: MCTensor, requires_grad=False, device=None, dtype=None) -> MCTensor:
    val = torch.zeros_like(input.as_subclass(Tensor), requires_grad=requires_grad, device=device, dtype=dtype)
    return MCTensor(val, nc=input.nc)

@implements(torch.ones_like)
def ones_like(input: MCTensor, requires_grad=False, device=None, dtype=None) -> MCTensor:
    val = torch.ones_like(input.as_subclass(Tensor), requires_grad=requires_grad, device=device, dtype=dtype)
    return MCTensor(val, nc=input.nc)

@implements(torch.clamp)
def clamp(input: MCTensor, min=None, max=None) -> MCTensor:
    return _clamp(input.tensor, min=min, max=max)

@implements(torch.clamp_min)
def clamp_min(input: MCTensor, min=None) -> MCTensor:
    return _clamp(input.tensor, min=min)

@implements(torch.clamp_max)
def clamp_max(input: MCTensor, max=None) -> MCTensor:
    return _clamp(input.tensor, max=max)

@implements(torch.norm)
def norm(input: MCTensor, dim=None, keepdim=False, p=2, **kw) -> MCTensor:
    return _norm(input.tensor, dim=dim, keepdim=keepdim, p=p)

@implements(torch.sum)
def sum(input: MCTensor, dim=None, keepdim=False, **kw) -> MCTensor:
    return _sum(input.tensor, dim=dim, keepdim=keepdim)

@implements(torch.mean)
def mean(input: MCTensor, dim=None, keepdim=False, **kw) -> MCTensor:
    return _mean(input.tensor, dim=dim, keepdim=keepdim)

@implements(torch.abs)
def abs(input: MCTensor) -> MCTensor:
    result_tensor = _Renormalize(input.tensor, input.nc)
    result_tensor = torch.abs(result_tensor) ## approximation, need to change, just take neg of these with negative first componet
    return result_tensor

@implements(torch.dot)
def dot(input: Union[MCTensor, int, float], other: Union[MCTensor, int, float]) -> MCTensor:
    if isinstance(input, MCTensor) and type(other) == Tensor:
        x = input
        y = other
    elif type(input) == Tensor and isinstance(other, MCTensor):
        x = other
        y = input
    else:
        ## implement dot product between mctensors
        raise NotImplemented
    return _Dot_MCN(x.tensor, y)

@implements(torch.mv)
def mv(input: Union[Tensor, MCTensor], other: Union[Tensor, MCTensor]) -> MCTensor:
    if input.dim() == 2 and other.dim() == 1:
        x = input  # matrix
        y = other  # vector
    elif input.dim() == 1 and other.dim() == 2:
        x = other  # matrix
        y = input  # vector
    else:
        raise NotImplemented
    if isinstance(x, MCTensor) and type(y) == Tensor:
        result = _MV_MC_T(x.tensor, y)
    elif type(x) == Tensor and isinstance(y, MCTensor):
        result = _MV_T_MC(x, y.tensor)
    else:
        ## implement mv between mctensors
        raise NotImplemented
    return result

@implements(torch.mm)
def mm(input: Union[Tensor, MCTensor], other: Union[Tensor, MCTensor]) -> MCTensor:
    if isinstance(input, MCTensor) and type(other) == Tensor:
        result = _MM_MC_T(input.tensor, other)
    elif type(input) == Tensor and isinstance(other, MCTensor):
        result = _MM_T_MC(input, other.tensor)
    else:
        ## implement mm between mctensors
        raise NotImplemented
    return result

@implements(torch.bmm)
def bmm(input: Union[Tensor, MCTensor], other: Union[Tensor, MCTensor]) -> MCTensor:
    if isinstance(input, MCTensor) and type(other) == Tensor:
        result, size, nc = _BMM_MC_T(input.tensor, other)
    elif type(input) == Tensor and isinstance(v, MCTensor):
        result, size, nc = _BMM_T_MC(input, other.tensor)
    else:
        ## implement mm between mctensors
        raise NotImplemented
    return result

@implements(torch.matmul)
def matmul(input: Union[MCTensor, Tensor], other: Union[MCTensor, Tensor]) -> MCTensor:
    x_dim, y_dim = input.dim(), other.dim()
    if x_dim == 1 and y_dim == 1:
        return dot(input, other)
    elif x_dim == 2 and y_dim == 2:
        return mm(input, other)
    elif (x_dim == 2 and y_dim == 1) or (x_dim == 1 and y_dim == 2):
        return mv(input, other)
    elif (x_dim > 2 and y_dim == 1) or (x_dim == 1 and y_dim > 2):
        return mul(input, other)
    elif x_dim == y_dim and x_dim == 3:
        if isinstance(input, MCTensor):
            result, size, nc = _BMM_MC_T(input.tensor, other)
        else:
            result, size, nc = _BMM_T_MC(input, other.tensor)
    elif x_dim == y_dim and x_dim == 4:
        if isinstance(input, MCTensor):
            result, size, nc = _4DMM_MC_T(input.tensor, other)
        else:
            result, size, nc = _4DMM_T_MC(input, other.tensor)
    elif x_dim > y_dim:
        y = other[(None,) * (x_dim - y_dim)]  # unsqueeze
        if x_dim == 3:
            if isinstance(input, MCTensor):
                result, size, nc = _BMM_MC_T(input.tensor, y)
            else:
                result, size, nc = _BMM_T_MC(input, y.tensor)
        elif x_dim == 4:
            if isinstance(input, MCTensor):
                result, size, nc = _4DMM_MC_T(input.tensor, y)
            else:
                result, size, nc = _4DMM_T_MC(input, y.tensor)
    elif x_dim < y_dim:
        x = input[(None,) * (y_dim - x_dim)]  # unsqueeze
        if y_dim == 3:
            if isinstance(input, MCTensor):
                result, size, nc = _BMM_MC_T(x.tensor, other)
            else:
                result, size, nc = _BMM_T_MC(x, other.tensor)
        elif y_dim == 4:
            if isinstance(input, MCTensor):
                result, size, nc = _4DMM_MC_T(x.tensor, other)
            else:
                result, size, nc = _4DMM_T_MC(x, other.tensor)
    else:
        ## implement mm between mctensors
        raise NotImplemented
    return result

@implements(torch.addmm)
def addmm(input: Union[MCTensor, Tensor], 
          mat1: Union[MCTensor, Tensor], 
          mat2: Union[MCTensor, Tensor], 
          beta=1.0, alpha=1.0) -> MCTensor:
    return beta * input + alpha * (mat1 @ mat2)

@implements(torch.reshape)
def reshape(input: MCTensor, shape) -> MCTensor:
    data = torch.reshape(input.as_subclass(Tensor), shape)
    extra_nc = input.res.shape[-1]
    res = torch.reshape(input.res.view(input.res.numel() // extra_nc, extra_nc), shape + (extra_nc,))
    return MCTensor.wrap_tensor_and_res_to_mctensor(data, res)

@implements(torch.nn.functional.relu)
def relu(input: MCTensor, inplace=False) -> MCTensor:
    val = torch.nn.functional.relu(input.as_subclass(Tensor), inplace=inplace)
    if inplace:
        input.res[input.as_subclass(Tensor) == 0] = 0
        return input
    else:
        res = input.res.clone()
        res[val == 0] = 0
        return MCTensor.wrap_tensor_and_res_to_mctensor(val, res)
    
@implements(torch.sigmoid)
def sigmoid(input) -> MCTensor:
    return 1/(torch.exp(-input) + 1)

@implements(torch.nn.functional.softmax)
def softmax(x: MCTensor, dim, *args, **kwargs) -> MCTensor:
    return _softmax(x.tensor, dim=dim)

@implements(torch.erf)
def erf(input: MCTensor) -> MCTensor:
    ### this is an approximation
    ret = torch.erf(input.as_subclass(Tensor))
    return MCTensor(ret, nc=input.nc)

@implements(torch.nn.functional.dropout)
def dropout(input: MCTensor, p=0.5, training=True, inplace=False) -> MCTensor:
    if training:
        val = torch.nn.functional.dropout(input.as_subclass(Tensor), p=p, training=True, inplace=inplace)
        if inplace:
            input.res[input.as_subclass(Tensor) == 0] = 0
            return input
        else:
            res = input.res.clone()
            res[val == 0] = 0
            return MCTensor.wrap_tensor_and_res_to_mctensor(val, res)    
    else:
        return input
    
@implements(torch.square)
def square(input: MCTensor) -> MCTensor:
    return _square(input.tensor)

@implements(torch.atanh)
def atanh(input: MCTensor) -> MCTensor:
    return _atanh(input.tensor)

@implements(torch.log1p)
def log1p(input: MCTensor) -> MCTensor:
    return _log1p_standard(input.tensor)

@implements(torch.nn.functional.linear)
def linear(input: Union[MCTensor, Tensor], weight: Union[MCTensor, Tensor], bias=None) -> MCTensor:
    if isinstance(input, MCTensor) and isinstance(weight, MCTensor):
        ## attention, here make input as tensor, as mul between MCTensors not supported yet
        input = input.as_subclass(Tensor)
    ret = torch.matmul(input, weight.T)
    if bias is None:
        return ret
    else:
        return ret + bias
    
@implements(torch.diag)
def diag(x: MCTensor, diagonal=0) -> MCTensor:
    indices_selected = torch.arange(x.numel(), dtype=torch.int64, device=x.device).view(*x.shape)
    selected_indices = torch.diag(indices_selected, diagonal=diagonal).view(-1)
    val = x.as_subclass(Tensor).view(-1)[selected_indices]
    res = x.res.view(x.numel(), x.res.shape[-1])[selected_indices]
    return MCTensor.wrap_tensor_and_res_to_mctensor(val, res)

@implements(torch.nn.functional.nll_loss)
def nll_loss(input: MCTensor, target: Tensor, **kw) -> MCTensor:
    return torch.mean(torch.diag(-input[:, target]))

@implements(torch.nn.functional.log_softmax)
def log_softmax(x: MCTensor, dim=None, **kw) -> MCTensor:
    return _log_softmax(x.tensor, dim=dim)

@implements(torch.nn.functional.cross_entropy)
def cross_entropy(x: MCTensor, target: Tensor, reduction='mean', label_smoothing=0.0, **kw) -> MCTensor:
    cross_entropy_x_tensor = _cross_entropy(x.tensor, target=target, reduction=reduction, label_smoothing=label_smoothing)
    return cross_entropy_x_tensor

@implements(torch.nn.functional.mse_loss)
def mse_loss(x: MCTensor, y: MCTensor, reduction='mean', **kw) -> MCTensor:
    return _mse_loss(x.tensor, y.tensor, reduction=reduction)

@implements(torch.sqrt)
def sqrt(input: MCTensor) -> MCTensor:
    return _sqrt(input.tensor)

@implements(torch.log)
def log(input: MCTensor) -> MCTensor:
    return _log(input.tensor)

@implements(torch.pow)
def pow(input: MCTensor, exponent: Union[Tensor, float, int]) -> MCTensor:
    return  _pow(input.tensor, exponent)

@implements(torch.exp)
def exp(input: MCTensor) -> MCTensor:
    return _exp(input.tensor)

@implements(torch.sinh)
def sinh(input: MCTensor) -> MCTensor:
    return _sinh(input.tensor)

@implements(torch.cosh)
def cosh(input: MCTensor) -> MCTensor:
    return _cosh(input.tensor)

@implements(torch.tanh)
def tanh(input: MCTensor) -> MCTensor:
    return _tanh(input.tensor)

@implements(torch.nn.functional.layer_norm)
def layer_norm(x: MCTensor, normalized_shape, weight=None, bias=None, eps=1e-05) -> MCTensor:
    nc = x.nc
    if isinstance(weight, torch.Tensor):
        mc_weight = torch.zeros(weight.size() + (nc,),
                                device=x.device, dtype=x.dtype)
        mc_weight[..., 0] = weight
    else:
        mc_weight = weight.tensor

    if isinstance(bias, torch.Tensor):
        mc_bias = torch.zeros(bias.size() + (nc,),
                              device=x.device, dtype=x.dtype)
        mc_bias[..., 0] = bias
    else:
        mc_bias = bias.tensor
    return _layer_norm(x.tensor, normalized_shape, mc_weight, mc_bias, eps=eps)