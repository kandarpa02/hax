import jax
import jax.numpy as jnp
import numpy as np
from .base import Module

# assumes your Module class (with _collect_params, _assign_params, etc.) already exists

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
        # Create a fresh module and record its parameters
        model = fn(*args, **kwargs)
        if not isinstance(model, Module):
            raise TypeError("The function must return a Module instance.")
        model._allow_call = True
        _ = model(jnp.zeros_like(args[0]))  # or example input to trigger param creation
        model._allow_call = False
        params = model._collect_params()
        return params, model  # return both, if needed

    def apply_fn(params, *args, **kwargs):
        # Create a new module and assign params
        model = fn(*args, **kwargs)
        model._assign_params(params)
        model._allow_call = True
        out = model(*args, **kwargs)
        model._allow_call = False
        return out

    return Transformed(init_fn, apply_fn)
