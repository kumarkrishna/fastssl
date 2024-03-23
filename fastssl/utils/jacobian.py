import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Callable, Union, Tuple
from functools import partial

def split_batch_gen(data_loader, batch_size, max_samples=0):
    """ Creates a generator that splits batches from data_loader into smaller batches
        of @batch_size and yields them.
        
        Note: For batches (img, label, img1, img2, ...) the generator discards any augmentations and labels, 
              and only returns the first item (img).
    """
    dl_batch_size = data_loader.batch_size
    assert batch_size <= dl_batch_size, f"Error: Jacobian batch size ({batch_size}) larger than data loader's batch size ({dl_batch_size})."

    if max_samples == 0:
        max_samples = len(data_loader) * dl_batch_size
    num_samples = 0

    # generator discards labels and augmentations
    for batch_id, batch in enumerate(data_loader):
        imgs, targets = batch[:2]
        for img, target in zip(
            torch.split(imgs, batch_size),
            torch.split(targets, batch_size)):
            
            yield img, target
            num_samples += 1
            if num_samples >= max_samples:
                return


""" Fast Jacobian computation routines
"""

def get_explicit_jacobian_fn(net, layer, data_loader):
    """Wrapper to initialize explicit Jacobian computation algorithm
    """
    activations = {}
    def hook_fn(m,i,o):
        activations["features"] = i[0]
    
    handle = None if layer is None else layer.register_forward_hook(hook_fn)

    device = next(net.parameters()).device
    batch = next(iter(data_loader))
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    batch = batch.to(device)
    output = net(batch)
    if layer is None:
        ndims = np.prod(output.shape[1:])
    else:
        ndims = np.prod(activations["features"].shape[1:])
    
    def tile_input(x):
        tile_shape = (ndims,) + (1,) * len(x.shape[1:])
        return x.repeat(tile_shape)
            
    def jacobian_fn(x):
        # discard augmentations
        inp = x[0] if isinstance(x, (list, tuple)) else x
        nsamples = inp.shape[0]
        inp = tile_input(inp)
        inp.requires_grad_(True)
        output = net(inp)
        features = output if layer is None else activations["features"]
        j = jacobian_features(inp, features, nsamples, ndims)
        inp.grad = None
        
        activations["features"] = None
        return j
    
    return jacobian_fn, handle


@torch.jit.script
def batched_matrix_vector_prod(u: torch.Tensor, J: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ Compute product u @ J.t() @ v
    """
    return torch.bmm(
        torch.transpose(
            torch.bmm(J, v), 
            1,
            2
        ), u
    ).squeeze(-1).squeeze(-1) # workaround to avoid squeezing batch dimension


@torch.jit.script
def spectral_norm_power_iteration(J: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute one iteration of the power method to estimate
        the largest singular value of J
    """
    u = torch.bmm(J, v)
    u /= torch.norm(u, p=2, dim=1).unsqueeze(-1)
    v = torch.matmul(torch.transpose(J, 1, 2), u)
    v /= torch.norm(v, p=2, dim=1).unsqueeze(-1)
    return (u, v)


@torch.jit.script
def spectral_norm(J: torch.Tensor, num_steps: int, atol: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Compute the spectral norm of @J using @num_steps iterations
        of the power method.
        
        @return u (torch.Tensor): left-singular vector
        @return sigma (torch.Tensor): largest singular value
        @return v (torch.Tensor): right-singular vector
    """
    device = J.device
    dtype = J.dtype
    J = J.view(J.shape[0], -1, J.shape[2])
    nbatches, nindims, noutdims = J.shape[0], J.shape[1], J.shape[2]
    
    batch_indices = torch.arange(nbatches, dtype=torch.long, device=device)
    atol = torch.full((1,), fill_value=atol, device=device, dtype=dtype)
    
    v = torch.randn(nbatches, noutdims, 1, device=device, dtype=dtype)
    v /= torch.norm(v, p=2, dim=1).unsqueeze(-1)
    sigma_prev = torch.zeros(nbatches, dtype=dtype, device=device)
    u_prev = torch.zeros((nbatches, nindims), dtype=dtype, device=device)
    v_prev = torch.zeros((nbatches, noutdims), dtype=dtype, device=device)
    
    for i in range(num_steps):
        u, v = spectral_norm_power_iteration(J, v)
        sigma = batched_matrix_vector_prod(u, J, v)
        diff_indices = torch.ge(
            torch.abs(sigma.squeeze() - sigma_prev[batch_indices]), atol
        )

        if not torch.any(diff_indices):
            break
        
        sigma_prev[batch_indices[diff_indices]] = sigma[diff_indices]
        u_prev[batch_indices[diff_indices]] = u[diff_indices].squeeze(-1)
        v_prev[batch_indices[diff_indices]] = v[diff_indices].squeeze(-1)
        u = u[diff_indices]
        v = v[diff_indices]
        J = J[diff_indices]
        batch_indices = batch_indices[diff_indices]
        
    return u_prev.squeeze(), sigma_prev, v_prev.squeeze()


@torch.jit.script
def jacobian_features(x: torch.Tensor, features: torch.Tensor, nsamples: int, ndims: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @features
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network features at @x
            
        Return:
            Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of feature dimensions
            of the network.
    """
    x.retain_grad()
    indexing_mask = torch.eye(ndims, device=x.device).repeat((nsamples,1))
    
    features.backward(gradient=indexing_mask, retain_graph=True)
    jacobian = x.grad.data.view(nsamples, ndims, -1).transpose(1,2)
    
    return jacobian


""" Low-memory Jacobian computation routines
"""

class Differentiable(nn.Module):
    """ Helper class to extract features from a specified layer
        wrapping the result in a nn.Module that can be turned
        into a pure function that is valid for torch.func
    """
    def __init__(self, net, layer):
        super(Differentiable, self).__init__()
        self.net = net
        self.activation = []
        def hook(module, input, output):
            self.activation.append(input[0])
            return output
        self.handle = layer.register_forward_hook(hook)

    def forward(self, x):
        self.activation = []
        out = self.net(x)
        return self.activation[0]
        
    def cleanup():
        if self.handle is not None:
            self.handle.remove()


def get_implicit_jacobian_fn(
    model: nn.Module,
    layer: Union[nn.Module, None],
    vmap: bool = True,
    detach_params: bool = True
) -> Tuple[Callable, Callable]:
    """ Set up implicit Jacobian computation
    
        Args:
            model: nn.Module - a neural network
            layer: nn.Module or None - if not None, the Jacobian of the layer's output w.r.t. input
                   to the network is computed
            vmap: bool - if True jvp_fn and vjp_fn are vmapped to support batched computation
            
        Returns:
            jacobian_fn: Callable with signature jacobian_fn(x: torch.Tensor) -> Tuple[Callable, Callable]
                         which sets up the implict Jacobian computation functions. The function returns:
                jvp_fn: Callable - a function with signature jvp_fn(u: torch.Tensor) -> torch.Tensor
                                   computing the push-forward of u using the Jacobian of @model_fn at @x
                vjp_fn: Callable - a function with signature vjp_fn(v: torch.Tensor) -> torch.Tensor
                                   computing the pull-back of v using the Jacobian of @model_fn at @x
                noutdims: int the codomain dimensionality of the Jacobian matrix
    """
    handle = None
    model_wrapped = model
    if layer is not None:
        model_wrapped = Differentiable(model, layer)
        handle = model_wrapped.handle
    
    if detach_params:
        params = {k: v.detach() for k, v in model_wrapped.named_parameters()}
    else:
        params = dict(model_wrapped.named_parameters())
    
    model_fn = lambda x: torch.func.functional_call(model_wrapped, params, (x,))
    if vmap:
        model_fn = torch.vmap(model_fn, in_dims=(0,), out_dims=(0,))
    
    def jacobian_fn(x):
        jvp_fn = lambda u: torch.func.jvp(model_fn, (x,), (u,))[1]
            
        (out, vjp_fn_) = torch.func.vjp(model_fn, x)
        vjp_fn = lambda v: vjp_fn_(v)[0]
        
        noutdims = np.prod(out.shape[1:])
        
        if vmap:
            jvp_fn = torch.vmap(jvp_fn, in_dims=(0,), out_dims=(0,))
            vjp_fn = torch.vmap(vjp_fn_, in_dims=(0,), out_dims=(0,))
        
        return jvp_fn, vjp_fn, noutdims
        
    return jacobian_fn, handle


#@partial(torch.vmap, in_dims=(0, None, None))
def spectral_norm_power_iteration_implicit(
    v: torch.Tensor,
    jvp_fn: Callable, 
    vjp_fn: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute one iteration of the power method to estimate
        the largest singular value of J, without instantiating J
        
        Args:
            v: torch.Tensor co-domain defined vector used to start the power iteration
            vjp_fn: Callable with signature vjp_fn(s) -> J.T @ s 
            jvp_fn: Callable with signature jvp_fn(s) -> J @ s
            
            where J is the Jacobian matrix implicitly computed by jvp_fn and vjp_fn
            
        Returns:
            u: torch.Tensor l2-normalized vector, representing the pull-back of v w.r.t. J
            t: torch.Tensor l2-normalized vector, rerpesenting the push-forward of u w.r.t J
            
            Equivalently, u = J.T @ v and t = J u = J @ (J.T @ v)
    """
    u = vjp_fn(v.unsqueeze(0))[0]
    u /= torch.norm(u, p=2, dim=-1).unsqueeze(-1)
    v = jvp_fn(u)
    v /= torch.norm(v, p=2, dim=-1).unsqueeze(-1)
    return (u, v)


#@torch.vmap
def spectral_norm_implicit(
    x: torch.Tensor,
    jacobian_fn: Callable,
    num_steps: int,
    atol: float = 1e-2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Compute the spectral norm of the model's Jacobian at @x using @num_steps iterations
        of the power method, without explicitly instantiating the Jacobian.
        
        Args:
            jacobian_fn: Callable a static function with signature jacobian_fn(x: torch.Tensor) -> Tuple[Callable, Callable]
                         with returns jvp and jvp computation routines.
            num_steps: int maximum number of power iterations
            atol: float sensitivity (stop when a power method iteration improves
                  the current estimate by at most @atol)
        
        @return u (torch.Tensor): left-singular vector
        @return sigma (torch.Tensor): largest singular value
        @return v (torch.Tensor): right-singular vector
    """
    device = x.device
    dtype = x.dtype
    nbatches, nindims = x.shape[0], np.prod(x.shape[1:])
    
    batch_indices = torch.arange(nbatches, dtype=torch.long, device=device)
    atol = torch.full((1,), fill_value=atol, device=device, dtype=dtype)
    
    jvp_fn, vjp_fn, noutdims = jacobian_fn(x)
    
    v = torch.randn(nbatches, noutdims, device=device, dtype=dtype)
    v /= torch.norm(v, p=2, dim=1).unsqueeze(-1)
    
    sigma_prev = torch.zeros(nbatches, dtype=dtype, device=device)
    u_prev = torch.zeros((nbatches, nindims), dtype=dtype, device=device)
    v_prev = torch.zeros((nbatches, noutdims), dtype=dtype, device=device)
    
    for i in range(num_steps):
        u, v = spectral_norm_power_iteration_implicit(v, jvp_fn, vjp_fn)
        sigma = torch.vmap(torch.dot)(
            vjp_fn(v)[0].squeeze(0),
            u.squeeze(0)
        )
        
        diff_indices = torch.ge(
            torch.abs(sigma.squeeze() - sigma_prev[batch_indices]), atol
        )

        if not torch.any(diff_indices):
            break
        
        sigma_prev[batch_indices[diff_indices]] = sigma[diff_indices]
        u_prev[batch_indices[diff_indices]] = u[diff_indices[None, :]]
        v_prev[batch_indices[diff_indices]] = v[diff_indices[None, :]]
        u = u[diff_indices[None, :]]
        v = v[diff_indices[None, :]]
        x = x[diff_indices]
        batch_indices = batch_indices[diff_indices]
        
        jvp_fn, vjp_fn, _ = jacobian_fn(x)
        
    return u_prev.squeeze(), sigma_prev, v_prev.squeeze()


""" Main entry-point
"""

def input_jacobian(net, layer, data_loader, batch_size=128, num_samples=0, use_cuda=False, bigmem=True):
    """ Compute average input Jacobian norm of features of @layer of @net using @data_loader.
    
        If bigmem = True, a fast method that explicity instantiates the Jacobian tensor
        is called, requiring O(NCHWK) memory and a single forward-backward pass (fast) through the network,
        where N is the batch size, (C, H, W) is the input data dimensionality, and K is the feature 
        space dimension.
        
        If bigmem = False, a slower implicit method is called, multiple forward-backward passes
        through the network (slow), but requring considerably less memory.
    
        Note: to control memory usage a @batch_size can be provided, in order for (potentially) 
              large batches of @dataloder to be broken down into smaller chunks.
              
        If num_samples > 0, the input Jacobian is estimated using num_samples training samples
    """  
    jacobian_norm = 0.
    num_samples = 0
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    
    dset_size = len(data_loader) * data_loader.batch_size
    num_batches = dset_size // batch_size + dset_size % batch_size
    progress_bar = tqdm(
        split_batch_gen(
            data_loader, batch_size, num_samples
        ),
        desc="Feature Input Jacobian",
        total = num_batches,
    )
    
    if bigmem:
        jacobian_fn, handle = get_explicit_jacobian_fn(net, layer, data_loader)
        def compute_operator_norm(x):
         jacobian = jacobian_fn(x)
         return spectral_norm(jacobian, num_steps=20)[1]
    else:
        jacobian_fn, handle = get_implicit_jacobian_fn(net, layer)
        compute_operator_norm = lambda x: spectral_norm_implicit(x, jacobian_fn=jacobian_fn, num_steps=10)[1]
    
    for i, batch in enumerate(progress_bar):
        
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        x = batch.to(device) # read input img from batch
        num_samples += x.shape[0]
        operator_norm = compute_operator_norm(x)
        
        jacobian_norm += operator_norm.sum().float().item()
        avg_norm = jacobian_norm / num_samples
        
        progress_bar.set_description(
            "Batch: [{}/{}] avg Jacobian norm: {:.2f}".format(
                i, num_batches -1, avg_norm
            )
        )
    if handle is not None:
        handle.remove()
    return avg_norm

