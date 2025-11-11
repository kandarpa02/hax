import jax
import jax.numpy as jnp
import numpy as np
from .base import Module


def _set_allow_call(module, value: bool):
    """Recursively enable/disable __call__ for a module and all its submodules."""
    module._allow_call = value
    for name, attr in module.__dict__.items():
        if isinstance(attr, Module):
            _set_allow_call(attr, value)


class Transformed:
    """Container for init and apply functions."""
    def __init__(self, init_fn, apply_fn):
        self.init = init_fn
        self.apply = apply_fn
    
    def __repr__(self) -> str:
        return f"TransformedFunction(\ninit:{self.init},\napply:{self.apply}\n)"

def transform(fn):
    """
    Turns an impure model-building function into pure (init, apply) pair.
    Works like haiku.transform.
    """
    def init_fn(rng, *args, **kwargs):
        if not isinstance(fn, Module):
            raise TypeError("The function must be a Module instance.")
        _set_allow_call(fn, True)
        fn.rng = rng
        fn._set_rng()
        dtype = None

        for k, v in kwargs.items():
            if k in ['dtype']:
                dtype = v
            else:
                dtype = args[0].dtype

        fn._set_dtype(dtype=dtype)
        
        _ = fn(*args)  # or example input to trigger param creation
        _set_allow_call(fn, False)
        params = fn._collect_params()
        return params 

    def apply_fn(rng, params, *args):
        # Create a new module and assign params
        fn._assign_params(params)
        _set_allow_call(fn, True)
        fn.rng = rng
        fn._set_rng()
        out = fn(*args)
        _set_allow_call(fn, False)
        return out

    return Transformed(init_fn, apply_fn)
