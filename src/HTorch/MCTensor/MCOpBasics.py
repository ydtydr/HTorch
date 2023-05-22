import math
import torch
import functools
import numpy as np
import itertools
import collections
import torch.sparse


def check_and_normalize_dim(x, dim):
    if dim is None: return None
    correct_dim = x.dim() - 1
    if type(dim) != int:
        raise RuntimeError(f"Expected an int dim but got {dim}")
    if dim >= 0:
        if dim >= correct_dim:
            raise IndexError(f'Dimension out of range (expected to be in range of [-{correct_dim}, {correct_dim-1}], but got {dim}')
    else:
        if dim < -correct_dim:
            raise IndexError(f'Dimension out of range (expected to be in range of [-{correct_dim}, {correct_dim-1}], but got {dim}')
        else:
            dim += correct_dim
    return dim

def num_to_mctensor_like(num, x):
    ret = torch.zeros(x.size(-1), dtype=x.dtype, device=x.device)
    ret[0] = num
    return ret

def fc_tensor_to_mctensor_like(tensor, x):
    ret = torch.zeros_like(x, device=x.device)
    ret[..., 0] = tensor
    return ret

def Two_Sum(a, b):
    """
    performs two-sum addition of two standard float vectors
    """
    x = a + b
    b_virtual = x - a
    a_virtual = x - b_virtual
    b_roundoff = b - b_virtual
    a_roundoff = a - a_virtual
    y = a_roundoff + b_roundoff
    return x, y

def QTwo_Sum(a, b):
    """
    performs quick-two-sum addition of two standard float vectors, work for n dimensional vectors, and b*n dimension
    """
    s = a + b
    e = b - (s - a)
    return s, e

def Split(a):
    """
    The following algorithm splits a 53-bit IEEE double precision floating point
    number into ahi and alo, each with 26 bits of significand,
    such that a = ahi + alo. ahi will contain the first 26 bits, while alo will contain the lower 26 bits.
    26 -> 12 for float iirc
    53 bit double and 24bit single, i.e. 26. and 12 for the constant
    """
    if a.dtype == torch.float16:
        constant = 6
    elif a.dtype == torch.float32:
        constant = 12
    elif a.dtype == torch.float64:
        constant = 26
    else:
        raise NotImplemented
    t = (2**constant+1)*a
    ahi = t-(t-a)
    alo = a - ahi
    return ahi, alo


def Two_Prod(a, b):
    """
    performs two-prod algorithm, computes p = fl(a × b) and e = err(a × b).
    """
    p = a * b
    ahi, alo = Split(a)
    bhi, blo = Split(b)
    e = ((ahi*bhi-p)+ahi*blo+alo*bhi)+alo*blo
    # """
    # performs two-prod-fma algorithm, computes p = fl(a × b) and e = err(a × b).
    # """
    # p = a * b
    # e = torch.addcmul(-p, a, b)
    return p, e

def Two_Div(a, b, nc=2):
    """
    performs two-div algorithm, computes q0 = fl(a / b) and q1 = err(a / b).
    """
    q = torch.zeros((*a.size(), nc), device=b.device, dtype=b.dtype)
    q[...,0] = a / b
    r = a
    for i in range(1, nc):
        approx_minus, e = Two_Prod(-q[..., i-1], b)
        r = (r + approx_minus) + e
        q[..., i] = r / b
    return q

def _Renormalize(x_tensor, r_nc=None):
    """
    x_tensor: tensor of a MCTensor to be renormalized
    r_nc: reduced number of components, <= nc of x
    """
    nc = x_tensor.size(-1)
    if r_nc is None:
        r_nc = nc - 1
    rough_x = torch.zeros_like(x_tensor)
    s = x_tensor.data[..., 0]  # the first (largest) component
    # first for loop, two sum from large to small components
    for i in range(1, nc):
        # two sum of s and i component
        s, rough_x_val = Two_Sum(x_tensor.data[..., i], s)
        ############# this could be addition/optional step 
#         nonzero_ind = torch.nonzero(rough_x_val, as_tuple=True)
#         nonzero_ind_x = tuple(
#             list(nonzero_ind) + [i * torch.ones(len(nonzero_ind[0])).long()])
#         # maybe needs to switch following behavior with copy, and make sure the data is changed
#         rough_x[nonzero_ind_x] = s[nonzero_ind]
#         s[nonzero_ind] = rough_x_val[nonzero_ind]
        ############# usual choice
        rough_x[..., i] = rough_x_val  # components in decreasing order now
    ############# only necessary when the first addition/optional step is used
#     _, indices = torch.sort(torch.abs(rough_x[..., 1:]), descending=True)
#     # fill this tensor in as contents for MCTensor
#     rough_x[..., 1:] = torch.gather(rough_x[..., 1:], -1, indices)
    #############
    # note here s is the largest component, there are nc-1 components in rough_x, with the first unfilled as 0
    normalized_x = torch.zeros_like(x_tensor.data)
    # second for loop
    for i in range(nc-1):
        s, e = Two_Sum(s, rough_x[..., -(i+1)])
        # the following aims to handle the if condition, but we may need to further discuss it
        nonzero_ind = torch.nonzero(e, as_tuple=True)
        nonzero_ind_x = tuple(
            list(nonzero_ind) + [-(1+i) * torch.ones(len(nonzero_ind[0]), dtype=torch.int64)])
        # maybe needs to switch following behavior with copy, and make sure the data is changed
        normalized_x[nonzero_ind_x] = s[nonzero_ind]
        s[nonzero_ind] = e[nonzero_ind]
    normalized_x[..., 0] = s
    # as of now, the components in normalized_x may not be correctly ordered, so we sort it according to
    # absolute values
    _, indices = torch.sort(torch.abs(normalized_x), descending=True)
    # fill this tensor in as contents for MCTensor
    normalized_x = torch.gather(normalized_x, -1, indices)[..., :r_nc]
    return normalized_x

def _Renormalize_opt(x_tensor, r_nc=None):
    """
    x_tensor: tensor of a MCTensor to be renormalized
    r_nc: reduced number of components, <= nc of x
    """
    nc = x_tensor.size(-1)
    if r_nc is None:
        r_nc = nc - 1
    rough_x = torch.zeros_like(x_tensor)
    s = x_tensor.data[..., 0]  # the first (largest) component
    # first for loop, two sum from large to small components
    for i in range(1, nc):
        # two sum of s and i component
        s, rough_x_val = Two_Sum(x_tensor.data[..., i], s)
        ############# this could be addition/optional step 
        nonzero_ind = torch.nonzero(rough_x_val, as_tuple=True)
        nonzero_ind_x = tuple(
            list(nonzero_ind) + [i * torch.ones(len(nonzero_ind[0]), dtype=torch.int64)])
        # maybe needs to switch following behavior with copy, and make sure the data is changed
        rough_x[nonzero_ind_x] = s[nonzero_ind]
        s[nonzero_ind] = rough_x_val[nonzero_ind]
        ############# usual choice
        # rough_x[..., i] = rough_x_val  # components in decreasing order now
    ############# only necessary when the first addition/optional step is used
    _, indices = torch.sort(torch.abs(rough_x[..., 1:]), descending=True)
    # fill this tensor in as contents for MCTensor
    rough_x[..., 1:] = torch.gather(rough_x[..., 1:], -1, indices)
    #############
    # note here s is the largest component, there are nc-1 components in rough_x, with the first unfilled as 0
    normalized_x = torch.zeros_like(x_tensor.data)
    # second for loop
    for i in range(nc-1):
        s, e = Two_Sum(s, rough_x[..., -(i+1)])
        # the following aims to handle the if condition, but we may need to further discuss it
        nonzero_ind = torch.nonzero(e, as_tuple=True)
        nonzero_ind_x = tuple(
            list(nonzero_ind) + [-(1+i) * torch.ones(len(nonzero_ind[0]), dtype=torch.int64)])
        # maybe needs to switch following behavior with copy, and make sure the data is changed
        normalized_x[nonzero_ind_x] = s[nonzero_ind]
        s[nonzero_ind] = e[nonzero_ind]
    normalized_x[..., 0] = s
    # as of now, the components in normalized_x may not be correctly ordered, so we sort it according to
    # absolute values
    _, indices = torch.sort(torch.abs(normalized_x), descending=True)
    # fill this tensor in as contents for MCTensor
    normalized_x = torch.gather(normalized_x, -1, indices)[..., :r_nc]
    return normalized_x


def _Simple_renormalize_old(tensor, r_nc):
#     tensor = torch.cat(
#         [tensor[0].data, tensor[1].data.unsqueeze(-1)], dim=-1)   # if tensor list is given
    _, indices = torch.sort(torch.abs(tensor.data), descending=True)
    # fill this tensor in as contents for MCTensor
    result = torch.gather(tensor.data, -1, indices)[..., :r_nc]
    return result


def _Simple_renormalize_copy(tensor_list, r_nc):
    raw_tensors = torch.cat([tensor_list[0].data, tensor_list[1].data.unsqueeze(-1)], dim=-1)
    actual_tensor = torch.zeros_like(raw_tensors[..., :r_nc])
    nnz_idx = torch.zeros(raw_tensors[..., 0].size(), dtype=torch.int64, device=raw_tensors.device)
    for i in range(raw_tensors.size(-1)):
        x_i = raw_tensors[..., i]
        cur_nnz_idx = (x_i != 0)
        actual_tensor[cur_nnz_idx, nnz_idx[cur_nnz_idx]] = x_i[cur_nnz_idx]
        nnz_idx[cur_nnz_idx] += 1
    return actual_tensor


def _Simple_renormalize_inplace(tensor_list, r_nc):
    tensors = torch.cat([tensor_list[0].data, tensor_list[1].data.unsqueeze(-1)], dim=-1)
    nc = tensors.size(-1)
    for i in range(nc):
        prev = tensors[..., i]
        for j in range(i + 1, nc):
            prev_is_0 = (prev == 0)
            if True not in prev_is_0: break
            cur = tensors[..., j]            
            prev[prev_is_0] = cur[prev_is_0]
            cur[prev_is_0] = 0
    return tensors[..., :r_nc]

def _Grow_ExpN(x_tensor, value, simple=True):
    nc = x_tensor.size(-1)
    Q = value
    h = torch.zeros_like(x_tensor)
    for i in range(1, nc+1):
        Q, hval = Two_Sum(x_tensor[..., -i], Q)
        if i == 1:
            last_tensor = hval.data
        else:
            h[..., -(i-1)] = hval.data
    h[..., 0] = Q
    # change from .unsqueeze(-1) to (*.shape, 1)
    if simple:
        h.data.copy_(_Simple_renormalize_old(torch.cat([h.data, last_tensor.data.view((*last_tensor.shape, 1))], dim=-1), r_nc=nc))
    else:
        h.data.copy_(_Renormalize(
            torch.cat([h.data, last_tensor.data.view((*last_tensor.shape, 1))], dim=-1), r_nc=nc))
    return h

def _AddMCN(x_tensor, y_tensor, simple=True):
    nc = x_tensor.size(-1)
    h = torch.zeros_like(x_tensor) if x_tensor.dim() >= y_tensor.dim() else torch.zeros_like(y_tensor)
    e = torch.tensor(0, dtype=x_tensor.dtype, device=x_tensor.device)  # since two_sum does the conversion to tensor
    hp_all, e1_all = Two_Sum(x_tensor, y_tensor)
    for i in range(nc):
        hi, e2 = Two_Sum(hp_all[..., i], e)
        h_to_append = hi if i == 0 else hi.data
        h[..., i] = h_to_append
        e = e1_all[..., i] + e2
    if simple:
        h.data.copy_(_Simple_renormalize_old(torch.cat([h.data, e.data.view((*e.shape, 1))], dim=-1), r_nc=nc))
    else:
        h.data.copy_(_Renormalize(torch.cat([h.data, e.data.view((*e.shape, 1))], dim=-1), r_nc=nc))
    return h

def _ScalingN(x_tensor, value, style='V', expand=False, simple=True):
    if style in ['T-MC', 'BMM-T-MC', '4DMM-T-MC']:
        nc = value.size(-1)
    else:
        nc = x_tensor.size(-1)
    e = torch.tensor(0)
    if style == 'V':
        hval_pre, e1 = Two_Prod(x_tensor, value.unsqueeze(-1))
    elif style == 'MC-T':
        hval_pre, e1 = Two_Prod(x_tensor.unsqueeze(-3), value.transpose(-1, -2).unsqueeze(-1).unsqueeze(0))
    elif style == 'T-MC':
        hval_pre, e1 = Two_Prod(x_tensor.unsqueeze(-2).unsqueeze(-1), value.transpose(-2, -3).unsqueeze(0))
    elif style == 'BMM-MC-T':
        # here
        hval_pre, e1 = Two_Prod(x_tensor.unsqueeze(-3), value.transpose(-1, -2).unsqueeze(1).unsqueeze(-1))
    elif style == 'BMM-T-MC':
        hval_pre, e1 = Two_Prod(x_tensor.unsqueeze(-2).unsqueeze(-1), value.unsqueeze(1).transpose(-2, -3))
    elif style == '4DMM-MC-T':
        hval_pre, e1 = Two_Prod(x_tensor.unsqueeze(-3), value.transpose(-1, -2).unsqueeze(2).unsqueeze(-1))
    elif style == '4DMM-T-MC':
        hval_pre, e1 = Two_Prod(x_tensor.unsqueeze(-2).unsqueeze(-1), value.unsqueeze(2).transpose(-2, -3))
    else:
        raise NotImplementedError()
    
    for i in range(nc):
        hval, e2 = Two_Sum(hval_pre[..., i], e)
        if i == 0:
            h_to_append = hval
            h = torch.zeros(h_to_append.size() + (nc,), dtype=h_to_append.dtype, device=h_to_append.device)
        else:
            h_to_append = hval.data
        h[..., i] = h_to_append
        e = e1[..., i] + e2
    if expand:
        r_nc = nc + 1
    else:
        r_nc = nc
    if simple:
        rh = _Simple_renormalize_old(torch.cat([h.data, e.data.view((*e.shape, 1))], dim=-1), r_nc=r_nc)
    else:
        rh = _Renormalize(torch.cat([h.data, e.data.view((*e.shape, 1))], dim=-1), r_nc=r_nc)
    h.data.copy_(rh.data[..., :nc])
    if expand:
        h = torch.cat([h, rh.data[..., -1:]], dim=-1)
    return h

def _DivMCN(x_tensor, y_tensor, case=0):
    nc = y_tensor.size(-1)
    h = torch.zeros_like(x_tensor) if x_tensor.dim() >= y_tensor.dim() else torch.zeros_like(y_tensor)
    # approx quotient q0
    q = x_tensor[..., 0] / y_tensor[..., 0]
    h[..., 0] = q
    for i in range(1, nc+1):
        r = _AddMCN(x_tensor, -_ScalingN(y_tensor, q), simple=True)  # r = x - y*q
        x_tensor = r
        q = x_tensor.data[..., 0] / y_tensor.data[..., 0]
        if i != nc:
            h[..., i] = q
    # if simple:
    #     h.data.copy_(_Simple_renormalize([h, q], r_nc=nc))
    # else:
    #     h.data.copy_(_Renormalize(
    #         torch.cat([h.data, q.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    if case==1:
        h.data.copy_(_Simple_renormalize_old(torch.cat([h.data, q.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    elif case==2:
        h.data.copy_(_Renormalize(torch.cat([h.data, q.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    elif case==3:
        h.data.copy_(_Renormalize_opt(torch.cat([h.data, q.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    return h

# def _DivMCN(x_tensor, y_tensor, case=0):
#     #case 0: w/o renorm; case 1: simple renorm; case 2: renorm; case 3: renorm_opt
#     print('case',  case)
#     nc = y_tensor.size(-1)   
#     x_tensor.data.copy_(_Renormalize(x_tensor.data, r_nc=nc))
#     y_tensor.data.copy_(_Renormalize(y_tensor.data, r_nc=nc))
#     qs = []
#     for i in range(nc): # can be extended to consider even higher order terms
#         h = Two_Div(x_tensor[..., 0], y_tensor[..., 0], nc=nc)
#         print('0 h', 0, h)
#         print('i x_tensor[..., i] e_inner in', i, x_tensor[..., i], e_inner)
#         r, e_inner = Two_Sum(x_tensor[..., i], e_inner)
#         print('i r e_inner out', i, r, e_inner)
#         for j in range(1, i+1):
#             tmp, e1 = Two_Prod(y_tensor[..., j], h[..., i - j])
#             print('i j r tmp', r, tmp)
#             r, e2 = Two_Sum(r, -tmp)
#             print('ij reduced r', r)
#             e_inner += -e1 + e2
#             print('ij e_inner e1 e2', e_inner, e1, e2, e_inner/y_tensor[..., 0])
# #         e_tmp = Two_Div(r, y_tensor[..., 0], nc=nc-i)
#         e_tmp = Two_Div(r, y_tensor[..., 0], nc=nc)
#         print('i e_tmp', i, e_tmp)
#         h = _AddMCN(h, e_tmp, simple=True)
# #         h[...,i:] = h[...,i:] + e_tmp
#         print('i h', i, h)
#     if case==1:
#         h.data.copy_(_Simple_renormalize(h.data, r_nc=nc))
#     elif case==2:
#         h.data.copy_(_Renormalize(h.data, r_nc=nc))
#     elif case==3:
#         h.data.copy_(_Renormalize_opt(h.data, r_nc=nc))
#     # print(f"h after renorm is:{h.data}")
#     return h


def _MultMCN(x_tensor, y_tensor, case=0):
    #case 0: w/o renorm; case 1: simple renorm; case 2: renorm; case 3: renorm_opt
    if y_tensor.dim() > 0:
        nc = y_tensor.size(-1)
    else:
        nc = x_tensor.size(-1)
    # cases where h is set to shape of y_tensor:
    # 1) x:size(3,1), y:size(3,5) -> x*y:size(3,5) #x smaller numel()
    # currently not support following sizes:
    # 1) x:size(3,1,5), y:size(3,5) -> x*y:size(3,3,5)
    # 2) x:size(3,5), y:size(3,1,5) -> x*y:size(3,3,5)  #x smaller dim(), TODO
    h = []
    e = 0.0
    for i in range(nc):
        r = e
        e = 0.0
        for j in range(i+1):
            # r += x_tensor[..., j] * y_tensor[..., i-j]
            tmp, e1 = Two_Prod(x_tensor[..., j], y_tensor[..., i-j])
            r, e2 = Two_Sum(r, tmp)
            e += e1 + e2
        h.append(r)
    h = torch.stack(h, dim=-1)
    if case==1:
        h.data.copy_(_Simple_renormalize_old(torch.cat([h.data, e.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    elif case==2:
        h.data.copy_(_Renormalize(torch.cat([h.data, e.data.unsqueeze(-1)], dim=-1), r_nc=nc))
    elif case==3:
        h.data.copy_(_Renormalize_opt(torch.cat([h.data, e.data.unsqueeze(-1)], dim=-1), r_nc=nc))        
    return h


# def _MultMCN(x_tensor, y_tensor, case=0):
#     # this might be faster, but worse error bounds
#     nc = x_tensor.size(-1)
#     ones_like_y = torch.ones_like(y_tensor)
#     ones_like_y[..., 1:].zero_()
#     y_inv = _DivMCN(ones_like_y, y_tensor, case=case)
#     result = _DivMCN(x_tensor, y_inv, case=case)
#     return result

# def _MultMCN(x_tensor, y_tensor, simple=True):
#     # this is slower, but with better error guarantee
#     nc = x_tensor.size(-1)
#     h = torch.zeros_like(x_tensor)
#     # approx quotient q0
#     p = x_tensor[..., 0] * y_tensor[..., 0]
#     h[...,0] = p
#     # convert p to MCTensor like
#     p_MC_tensor = torch.zeros(p.size()+(nc,)).to(x_tensor)
#     p_MC_tensor[..., 0] = p.data
#     for i in range(1, nc+1):
#         # if replace this with an algo similar to scalingN, this would be much faster
#         minus_pdivy = - _DivMCN(p_MC_tensor, y_tensor, simple=simple)
#         e = _AddMCN(x_tensor, minus_pdivy, simple=simple)
#         x_tensor = e
#         p = x_tensor.data[..., 0] * y_tensor.data[..., 0]
#         if i!=nc: h[..., i] = p
#         p_MC_tensor.zero_()
#         p_MC_tensor[..., 0] = p.data
#     if simple:
#         h.data.copy_(_Simple_renormalize([h, p], r_nc = nc))
#     else:
#         h.data.copy_(_Renormalize(torch.cat([h.data, p.data.unsqueeze(-1)], dim=-1), r_nc = nc))
#     return h


# def _exp(input_tensor):
#     nc = input_tensor.size(-1)
#     MCF_exp = torch.exp(input_tensor)
#     tmp = MCF_exp[..., 0:1]
#     for i in range(1, nc):
#         tmp = _ScalingN(tmp, MCF_exp[..., i], expand=True)
#     return tmp


def _square(input_tensor):
    return _MultMCN(input_tensor, input_tensor)
    # nc = input_tensor.size(-1)
    # x1 = input_tensor[..., 0]
    # if nc == 1:
    #     return torch.square(x1)
    # tmp = torch.zeros_like(input_tensor)
    # x2 = input_tensor[..., 1]
    # tmp[..., 0] = torch.square(x1)
    # return _Grow_ExpN(tmp, (2*x1*x2).data)


def _nth_root(input_tensor, n):
    # x_{i + 1} = x_i + x_i * (x_i * (1 - a * x_i^n)) / n
    x = torch.zeros_like(input_tensor)
    x[..., 0].copy_(torch.pow(input_tensor.sum(-1), -1/n)) # warm start
    one = torch.zeros(input_tensor.size(-1), dtype=input_tensor.dtype, device=input_tensor.device)
    one[0] = 1
    for _ in range(2):
        x = _AddMCN(x, _MultMCN(x, _AddMCN(-_MultMCN(input_tensor, _pow(x, n)), one)) / n)
    x1 = torch.zeros_like(input_tensor)
    x1[..., 0] = 1
    return _DivMCN(x1, x) # 1 / x as the return 


# no need to replace recursion with while-loop. The stack should be shallow
def _pow(x, exponent):
    if exponent == 1:
        return x.clone()
    if exponent > 1 and int(exponent) != exponent:
        return _MultMCN(_pow(x, exponent - int(exponent)), _pow(x, int(exponent)), case=3)
    if exponent < 0:
        one = torch.zeros_like(x)
        one[..., 0] = 1.0
        return _DivMCN(one, _pow(x, -exponent))
    if 0 < exponent < 1 and torch.floor(torch.as_tensor(1/exponent)) == int(1/exponent):
        return _nth_root(x, 1/exponent)
    if exponent < 1:
        # base case. the difference are small so we use pytorch power
        ret = torch.zeros_like(x)
        ret[..., 0] = torch.pow(x.sum(-1), exponent)
        return ret 
    x_sq = _MultMCN(x, x)
    if exponent % 2 == 0:
        return _pow(x_sq, exponent / 2)
    else:
        return _MultMCN(x, _pow(x_sq, (exponent - 1) / 2))


def _sqrt(input_tensor):
    # x_{i + 1} = x_i + (x_i * (1 - a * x_i^2)) / 2
    x = torch.zeros_like(input_tensor)
    x[..., 0].copy_(1 / torch.sqrt(input_tensor[..., 0])) # warm start
    for _ in range(2):
        x_sq = _MultMCN(x, x)
        x += x * (1 - _MultMCN(input_tensor, x_sq)) / 2
    return _MultMCN(x, input_tensor) # a * x as the return 


def _reduce_exp(x):
    # MPFR paper explicit computes and stores a high-precision ln2 value, but here we use the heuristic
    k = 1024
    ln2 = math.log(2) 
    ln2_mc_tensor = num_to_mctensor_like(ln2, x)
    invln2_mc_tensor = _DivMCN(num_to_mctensor_like(1.0, x), ln2_mc_tensor)
    m = torch.floor(torch.add(0.5, x, alpha=invln2_mc_tensor[..., 0])) # ⌊ x/log(2) + 1/2 ⌋
    kr = _AddMCN(x, -_MultMCN(m, ln2_mc_tensor), simple=False) # x - m * ln2
    r = torch.ldexp(kr, torch.tensor(-10, device=x.device)) 
    return k, m, r

# r -> kr
def _exp_taylor(r, n=5): 
    ret = num_to_mctensor_like(1.0, r)
    for i in range(1, n + 1):
        denom = num_to_mctensor_like(math.factorial(i), r)
        ret = _AddMCN(_DivMCN(_pow(r, i), denom), ret, simple=False)
    return ret

def _exp(input_tensor):
    k, m, r = _reduce_exp(input_tensor)
    return torch.ldexp(_pow(_exp_taylor(r), k), m)
    

def _log(input_tensor, n=10):
    # xk = xk - 1 + a exp^(-xk)
    x = torch.zeros_like(input_tensor) # warm start
    # check the magnitude
    x[..., 0] = torch.log(input_tensor[..., 0]) 
    minus1 = num_to_mctensor_like(-1, x)
    for _ in range(n):
        e_x = _exp(x)
        x = _AddMCN(_AddMCN(x, minus1), _DivMCN(input_tensor, e_x))
    return x


def _sinh(x):
    # (ex - e^{-x}) / 2, or (e^{2x} - 1) / (2ex)
    nom = _AddMCN(_exp(x), - _exp(-x))
    return torch.ldexp(nom, torch.tensor(-1, device=nom.device))


def _cosh(x):
    # (ex + e^{-x}) / 2, or (e^{2x} + 1) / (2ex)
    nom = _AddMCN(_exp(x), _exp(-x))
    return torch.ldexp(nom, torch.tensor(-1, device=nom.device))


def _tanh_small(x, t):
    g = _square(x)
    # x + x R(X)
    if t <= 24:
        p0 = -0.8237728127
        p1 = num_to_mctensor_like(-0.003831010665, x)
        q0 = 2.471319654
        Pg = _Grow_ExpN(_MultMCN(g, p1), p0)
        Rx = _DivMCN(_MultMCN(g, Pg), _Grow_ExpN(g, q0)) 
    else:
        p0 = -1613.4119023996228053
        p1 = -99.225929672236083313
        p2 = num_to_mctensor_like(-0.96437492777225469787, x)
        q0 = 4840.2357071988688686
        q1 = 2233.7720718962312926
        q2 = 112.74474380534949335
        # (p2 * g + p1) * g + p0
        nom = _Grow_ExpN(_MultMCN(g, _Grow_ExpN(_MultMCN(g, p2), p1)), p0)
        # ((g + q2) * g + q1) * g + q0
        denom = _Grow_ExpN(_MultMCN(g, _Grow_ExpN(_MultMCN(g, _Grow_ExpN(g, q2)), q1)), q0)
        Rx = _MultMCN(g, _DivMCN(nom, denom))
    return _AddMCN(x, _MultMCN(x, Rx))

def _tanh_medium(x):
    # 2 * (0.5 - 1/(1-e^{2x}))
    ex2 = _exp(torch.ldexp(x, torch.tensor(1, dtype=x.dtype, device=x.device)))
    one = num_to_mctensor_like(1.0, x)
    one_div_one_plus_ex2 = _DivMCN(one, _Grow_ExpN(ex2, 1.0))
    return torch.ldexp(_Grow_ExpN(-one_div_one_plus_ex2, 0.5), torch.tensor(1, dtype=x.dtype, device=x.device))

# reference: https://www.math.utah.edu/~beebe/software/ieee/tanh.pdf 
def _tanh(x):
    t = { # fractional bit
        torch.float16: 10,
        torch.float32: 23,
        torch.float64: 52
    }[x.dtype]
    small_breakpoint = math.sqrt(3) * 2 ** ((-t-1)/2)
    medium_breakpoint = math.log(3) / 2
    large_breakpoint = (t + 2) * math.log(2) / 2
    abs_x =  torch.abs(x[..., 0])
    
    less_than_small = abs_x <= small_breakpoint
    small_to_medium = ((small_breakpoint < abs_x) & (abs_x <= medium_breakpoint))
    medium_to_large = ((medium_breakpoint < abs_x) & (abs_x <= large_breakpoint))
    larger_than_large = abs_x > large_breakpoint

    ret = torch.zeros_like(x)
    ret[less_than_small] = x[less_than_small] # tanh x \sim x
    ret[small_to_medium] = _tanh_small(x[small_to_medium], t=t)
    ret[medium_to_large] = _tanh_medium(x[medium_to_large])
    ret[larger_than_large] = torch.sign(x[larger_than_large])

    return ret

# https://git.musl-libc.org/cgit/musl/tree/src/math/log1p.c
def _log1p_musl_simple(x):
    ret = torch.zeros_like(x, device=x.device)
    x_nnz = x[..., 0] != 0
    u = _Grow_ExpN(x[x_nnz], 1.0)
    ret[x_nnz] = _MultMCN(_log(u), _DivMCN(x[x_nnz], _Grow_ExpN(u, -1.0)))
    return ret

# when x is really close to 0 (abs(x) <= 1e-4), we use taylor series
def _log1p_taylor_series(x):
    # log(1 + x) = x - x2/2 + x3/3 - x4/4 + x5/5 - x6/6 + x7/7...
    x2 = _square(x)
    x3 = _MultMCN(x2, x) # sufficient for float16
    x4 = _MultMCN(x3, x) 
    x5 = _MultMCN(x4, x) # sufficient for float32
    x6 = _MultMCN(x5, x) 
    x7 = _MultMCN(x6, x) # 1e^-28, even sufficient for float64
    half_x2 = torch.ldexp(x2, torch.tensor(-1, device=x.device))
    one_third_x3 = _MultMCN(x3, num_to_mctensor_like(1/3, x3))
    one_fourth_x4 = torch.ldexp(x4, torch.tensor(-2, device=x.device))
    one_fifth_x5 = _MultMCN(x5, num_to_mctensor_like(1/5, x5))
    one_sixth_x6 = _MultMCN(x6, num_to_mctensor_like(1/6, x6))
    one_seventh_x7 = _MultMCN(x7, num_to_mctensor_like(1/7, x7))
    first_three_terms = _AddMCN(_AddMCN(x, -half_x2), one_third_x3)
    last_four_terms = _AddMCN(_AddMCN(_AddMCN(-one_fourth_x4, one_fifth_x5), -one_sixth_x6), one_seventh_x7)
    return _AddMCN(first_three_terms, last_four_terms)

# 1. find k and f such that
#         1+x = 2^k * (1+f),
#         where  sqrt(2)/2 < 1+f < sqrt(2) 
def _log1p_normal(x):
    u = _Grow_ExpN(x, 1.0)
    c = _AddMCN(u, -u)
    k = torch.floor(torch.add(torch.log2(u[..., 0]), 0.5))
    k_mc_tensor = k.view(*k.shape, 1) * torch.ones_like(u)
    f_plus_1 = torch.ldexp(u, -k_mc_tensor)
    f_plus_1_abs = torch.abs(f_plus_1)
    log1p_f_plus_1 = torch.zeros_like(f_plus_1)
    large_region = (f_plus_1_abs[..., 0] > 1.1e-4) | (f_plus_1_abs[..., 0] < (1 - 1e-4))
    log1p_f_plus_1[large_region] = _log(f_plus_1[large_region])
    log1p_f_plus_1[~large_region] = _log1p_taylor_series(_Grow_ExpN(f_plus_1[~large_region], -1.0))
    log2 = math.log(2)
    k_mc_tensor = torch.zeros_like(x)
    k_mc_tensor[..., 0] = k
    return _AddMCN(_AddMCN(log1p_f_plus_1, _MultMCN(k_mc_tensor, num_to_mctensor_like(log2, k_mc_tensor))), _DivMCN(c, u))


def _log1p_small_inplace_(x, absx, ret, tiny_line):
    ret[(absx <= tiny_line)] = x[(absx <= tiny_line)]
    ret[(tiny_line <= absx)] = _log1p_taylor_series(x[(tiny_line <= absx)])
    return ret

# https://git.musl-libc.org/cgit/musl/tree/src/math/log1p.c
# 1. find k and f such that
#         1+x = 2^k * (1+f),
#         where  sqrt(2)/2 < 1+f < sqrt(2) 
# 2. approximate log1p(f) by taylor expansion
# 3. log1p(x) = k*ln2 + log(1+f) + c/u,
#       in which c = (1+x)-u, u = 1 + x
def _log1p_standard(x: torch.Tensor):
    t = { # fractional bit + exponent
        torch.float16: 10 + 5,
        torch.float32: 23 + 8,
        torch.float64: 52 + 11
    }[x.dtype]
    tiny_line = 2 ** -t
    ret = torch.zeros_like(x, device=x.device)
    abs_x = torch.abs(x[..., 0])
    small_region = (abs_x <= 1e-4)
    normal_region = ~small_region
    _log1p_small_inplace_(x[small_region], abs_x[small_region], ret[small_region], tiny_line)
    ret[normal_region] = _log1p_normal(x[normal_region])
    ret[x < -1] = torch.nan
    ret[x == -1] = -torch.inf
    return ret


def _clamp(x: torch.Tensor, min=None, max=None):
    ret = x.clone()
    if min is not None:
        smaller_region = x[..., 0] <= min
        ret[..., 0][smaller_region] = min
        ret[..., 1:][smaller_region] = 0
    if max is not None:
        larger_region = x[..., 0] >= max
        ret[..., 0][larger_region] = max
        ret[..., 1:][larger_region] = 0
    return ret


def _sum(x: torch.Tensor, dim=None, keepdim=False):
    dim = check_and_normalize_dim(x, dim)
    size_x, nc = x.size()[:-1], x.size(-1)
    if dim is None:
        x_view = x.view(torch.prod(torch.tensor(size_x)), nc)
        ret = x_view[0]
        for vec in x_view[1:]:
            ret = _AddMCN(ret, vec)
        if keepdim:
            return ret.view(*tuple(1 for _ in range(x.dim() - 1)) + (x.size(-1),))
        else:
            return ret
    else:
        ret = x.index_select(dim, torch.tensor(0, device=x.device))
        for i in range(1, x.size(dim)):
            vec = x.index_select(dim, torch.tensor(i, device=x.device))
            ret = _AddMCN(ret, vec)
        if keepdim:
            return ret
        else:
            return ret.squeeze(dim)

def _mean(x: torch.Tensor, dim=None, keepdim=False):
    dim = check_and_normalize_dim(x, dim)
    size_x, nc = x.size()[:-1], x.size(-1)
    if dim is None:
        x_view = x.view(torch.prod(torch.tensor(size_x)), nc)
        ret = x_view[0]
        for vec in x_view[1:]:
            ret = _AddMCN(ret, vec)
        denom = torch.prod(torch.tensor(size_x))
        denom = num_to_mctensor_like(denom.item(), ret)
        ret = _DivMCN(ret, denom)
        if keepdim:
            return ret.view(*tuple(1 for _ in range(x.dim() - 1)) + (x.size(-1),))
        else:
            return ret
    else:
        ret = x.index_select(dim, torch.tensor(0, device=x.device))
        for i in range(1, x.size(dim)):
            vec = x.index_select(dim, torch.tensor(i, device=x.device))
            ret = _AddMCN(ret, vec)
            if (i+1) == x.size(dim):
                denom = num_to_mctensor_like(x.size(dim), ret)
                ret = _DivMCN(ret, denom)
        if keepdim:
            return ret
        else:
            return ret.squeeze(dim)
        
        
def _abs(x):
    ret = _Renormalize(x, x.size(-1))
    neg_region = x[..., 0] < 0
    ret[neg_region] = -ret[neg_region]
    return ret


def _norm(x: torch.Tensor, dim=None, keepdim=False, p=2):
    dim = check_and_normalize_dim(x, dim)
    if p == float('inf'):
        abs_x = _abs(x)
        if dim is None:
            res = _exact_max(abs_x)
        else:
            res = _approx_max(abs_x, dim=dim).values
        if keepdim:
            if dim is None:
                return res.view(*tuple(1 for _ in range(x.dim() - 1)) + (x.size(-1),))
            else:
                return res.unsqueeze(dim=dim)
        else:
            return res
    elif p == 2: # special L2 norm
        return _sqrt(_sum(_square(x), dim=dim, keepdim=keepdim))
    elif (type(p) == float) or (type(p) == int) or \
        (
            isinstance(p, torch.Tensor) and p.dtype in [
            torch.float16, torch.float32, torch.float64,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, 
        ]): # general case
        abs_x = _abs(x)
        if p == 1:
            summed_res = _sum(abs_x, dim=dim, keepdim=keepdim)
        else:
            summed_res = _sum(_pow(abs_x, p), dim=dim, keepdim=keepdim)
        if summed_res.dim() == 1:
            summed_res = summed_res.view(1, summed_res.size(-1))
        res = _pow(summed_res, (1/p)) if p != 1 else summed_res
        if keepdim:
            return res
        else:
            return res.squeeze()
    else:
        raise NotImplementedError()



class MCTensorlike_Atanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor): # x is a MCTensor-like tensor
        x = _clamp(x, min=-1 + 1e-15, max=1 - 1e-15)
        ctx.save_for_backward(x)
        log1p_x_minus_log1p_minus_x = _AddMCN(_log1p_standard(x), -_log1p_standard(-x))
        return torch.ldexp(log1p_x_minus_log1p_minus_x, torch.tensor(-1, device=x.device))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return _DivMCN(grad_output, (1 - _square(input)))

def _atanh(x):
    return MCTensorlike_Atanh.apply(x)

def _exact_max(x):
    nc = x.size(-1)
    x_renormalize = _Renormalize(x, r_nc=nc)
    x_renormalize_contiguous = x_renormalize.view(torch.prod(torch.tensor(x_renormalize.shape[:-1])), nc)
    max_value_nc_1 = torch.max(x_renormalize_contiguous[:, 0])
    max_idx_nc_1 = torch.argwhere(x_renormalize_contiguous[:, 0] == max_value_nc_1)
    if len(max_idx_nc_1) == 1:
        return x_renormalize_contiguous[max_idx_nc_1].view(1, nc)
    else:
        max_idx_nc_prev = max_idx_nc_1
        for i in range(1, x.size(-1)):
            max_value_nc_i = torch.max(x_renormalize_contiguous[max_idx_nc_prev, i])
            max_idx_nc_i = torch.argwhere(x_renormalize_contiguous[max_idx_nc_prev, i].view(-1) == max_value_nc_i)
            if len(max_idx_nc_i) == 1:
                return x_renormalize_contiguous[max_idx_nc_prev[max_idx_nc_i.view(-1)]].view(1, nc)
            else:
                max_idx_nc_prev = max_idx_nc_prev[max_idx_nc_i].squeeze(-1)
        # elts equal for all nc, and we choose the first one
        return x_renormalize_contiguous[max_idx_nc_prev[max_idx_nc_i][0]].view(1, nc)

def _approx_max(x, dim=None):
    size = x.size()[:-1]
    if dim is None:
        dim = 0
    elif dim == len(size):
        raise NotImplementedError(f"dim is out of range of [{-len(size)}, {len(size) - 1}]")
    elif dim < 0:
        dim = len(size) + dim
    return torch.max(x, dim=dim)


def _exact_min(x):
    nc = x.size(-1)
    x_renormalize = _Renormalize(x, r_nc=nc)
    x_renormalize_contiguous = x_renormalize.view(torch.prod(torch.tensor(x_renormalize.shape[:-1])), nc)
    min_value_nc_1 = torch.min(x_renormalize_contiguous[:, 0])
    min_idx_nc_1 = torch.argwhere(x_renormalize_contiguous[:, 0] == min_value_nc_1)
    if len(min_idx_nc_1) == 1:
        return x_renormalize_contiguous[min_idx_nc_1].view(1, nc)
    else:
        min_idx_nc_prev = min_idx_nc_1
        for i in range(1, x.size(-1)):
            min_value_nc_i = torch.min(x_renormalize_contiguous[min_idx_nc_prev, i])
            min_idx_nc_i = torch.argwhere(x_renormalize_contiguous[min_idx_nc_prev, i].view(-1) == min_value_nc_i)
            if len(min_idx_nc_i) == 1:
                return x_renormalize_contiguous[min_idx_nc_prev[min_idx_nc_i.view(-1)]].view(1, nc)
            else:
                min_idx_nc_prev = min_idx_nc_prev[min_idx_nc_i].squeeze(-1)
        # elts equal for all nc, and we choose the first one
        return x_renormalize_contiguous[min_idx_nc_prev[min_idx_nc_i][0]].view(1, nc)


def _approx_min(x, dim=None):
    size = x.size()[:-1]
    if dim is None:
        dim = 0
    elif dim == len(size):
        raise NotImplementedError(f"dim is out of range of [{-len(size)}, {len(size) - 1}]")
    elif dim < 0:
        dim = len(size) + dim
    return torch.min(x, dim=dim)


def _softmax(x, dim=None):
    size, nc = x.size()[:-1], x.size(-1)
    if dim is None:
        raise NotImplementedError()
    elif dim == len(size):
        raise NotImplementedError(f"dim is out of range of [{-len(size)}, {len(size) - 1}]")
    elif dim < 0:
        dim = len(size) + dim

    multi_dim_expand_size = tuple(1 for _ in size[:dim]) + (size[dim],) + tuple(1 for _ in size[dim+1:]) + (1,)
    single_dim_view_size = size[:dim] + (1,) + size[dim+1:] + (nc,)

    reduced_x = _AddMCN(x, - _approx_max(x, dim=dim).values.view(single_dim_view_size).repeat(multi_dim_expand_size))
    exp_reduced_x = _exp(reduced_x)
    exp_reduced_x[torch.isinf(exp_reduced_x)] = 1.0
    exp_reduced_x[torch.isnan(exp_reduced_x)] = 0.0
    # exclude nc dimension

    # exp(x - max(x)) / sum(exp(x - max(x)))
    numerator = exp_reduced_x
    denominator = exp_reduced_x.select(dim, 0)
    for i in range(1, size[dim]):
        denominator = _AddMCN(denominator, exp_reduced_x.select(dim, i))
    
    ret = torch.zeros_like(numerator)
    num_flatten = numerator.view(torch.prod(torch.as_tensor(size)), nc)
    nonzero_num_idx = num_flatten[:, 0].nonzero()
    nonzero_num = num_flatten[nonzero_num_idx, :]

    nonzero_denom = denominator.view(single_dim_view_size).repeat(multi_dim_expand_size).view(num_flatten.shape)
    nonzero_denom = nonzero_denom[nonzero_num_idx, :]
    ret.view(num_flatten.shape)[nonzero_num_idx] = _DivMCN(nonzero_num, nonzero_denom)
    return ret


def _log_softmax(x, dim=None):
    size, nc = x.size()[:-1], x.size(-1)
    if dim is None:
        raise NotImplementedError()
    elif dim == len(size):
        raise NotImplementedError(f"dim is out of range of [{-len(size)}, {len(size) - 1}]")
    elif dim < 0:
        dim = len(size) + dim

    multi_dim_expand_size = tuple(1 for _ in size[:dim]) + (size[dim],) + tuple(1 for _ in size[dim+1:]) + (1,)
    single_dim_view_size = size[:dim] + (1,) + size[dim+1:] + (nc,)

    # (x - max(x)) - np.log(np.sum(np.exp(x - max(x))))
    reduced_x = _AddMCN(x, - _approx_max(x, dim=dim).values.view(single_dim_view_size).repeat(multi_dim_expand_size))
    exp_reduced_x = _exp(reduced_x)
    exp_reduced_x[torch.isinf(exp_reduced_x)] = 1.0
    exp_reduced_x[torch.isnan(exp_reduced_x)] = 0.0
    reduced_sum_exp_x = exp_reduced_x.select(dim, 0)
    for i in range(1, size[dim]):
        reduced_sum_exp_x = _AddMCN(reduced_sum_exp_x, exp_reduced_x.select(dim, i))
    
    return _AddMCN(reduced_x, -_log(reduced_sum_exp_x.view(single_dim_view_size).repeat(multi_dim_expand_size)))


# logits: [N, C, nc]    float
# target: [N, C, nc]    float
# target: [N]           int64
# target: [N, C]        float
def _cross_entropy(mc_logits, target, reduction='mean', label_smoothing=0.0):
    if mc_logits.dim() != 3 or target.dim() > 2:
        raise NotImplementedError()

    N, C, nc = mc_logits.shape
    L = _log_softmax(mc_logits, dim=1)
    
    Y = torch.zeros_like(mc_logits)
    Y[np.arange(N), target, 0] = 1.0

    if label_smoothing != 0.0:
        Y = (1 - label_smoothing) * Y + label_smoothing / C
    
    LY = - _MultMCN(L, Y)
    l = LY[0]
    for i in range(1, N):
        l = _AddMCN(l, LY[i]) 

    summed_l = l[0]
    for i in range(1, C):
        summed_l = _AddMCN(summed_l, l[i])
    l = summed_l 

    if reduction == 'mean':
        return _DivMCN(l,  num_to_mctensor_like(N, mc_logits))
    elif reduction == 'sum':
        return l
    else:
        raise NotImplementedError()


def _mse_loss(x, y, reduction='mean'):
    x = x.view(x.shape[0], int(torch.prod(torch.tensor(x.shape[1:-1]))), x.shape[-1])
    y = y.view(y.shape[0], int(torch.prod(torch.tensor(y.shape[1:-1]))), y.shape[-1])

    N, d, nc = x.shape

    l = _square(_AddMCN(x[0], -y[0]))
    for i in range(1, N):
        l = _AddMCN(l, _square(_AddMCN(x[i], -y[i])))
    
    l = l.view(*l.shape[:-1], nc)
    aggregated_l = l[0]
    for li in l[1:]:
        aggregated_l = _AddMCN(aggregated_l, li)
    
    if reduction == 'sum':
        return aggregated_l
    elif reduction == 'mean':
        denom_nc = num_to_mctensor_like(N * d, aggregated_l)
        return _DivMCN(aggregated_l, denom_nc)
    else:
        raise NotImplementedError()


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-05):
    dims = [-(i+2) for i in range(len(normalized_shape))]
    mean = x.mean(dim=dims, keepdim=True)
    mean_x2 = _square(x).mean(dim=dims, keepdim=True)
    var = _AddMCN(mean_x2, -_square(mean))
    mc_eps = torch.zeros(x.size(-1), dtype=x.dtype, device=x.device)
    mc_eps[0] = eps
    x_norm = _DivMCN(_AddMCN(x, -mean), _sqrt(_AddMCN(var, mc_eps)))
    if weight is not None:
        x_norm = _AddMCN(_MultMCN(x_norm, weight), bias)
    return x_norm


# https://github.com/scipy/scipy/blob/main/scipy/special/cdflib/erf.f
# def _erf(x, n=5):
#     ret = torch.zeros_like(x)
#     x_greater_1_part = torch.where(x[..., 0] > 1)
#     x_less_1_part = ~x_greater_1_part
#     pi_sqrt = math.sqrt(math.pi)
    
#     exp_x_neg_sq = _exp(-_square(x))
#     x_greater_1_value = 
#     for i in range(n):
#         x[x_greater_1_part] = ((-1) ** i) * (2 * i - 1)