from .base import Module

def _set_allow_call(module, value: bool):
    """Recursively enable/disable __call__ for a module and all its submodules."""
    module._allow_call = value
    for name, attr in module.__dict__.items():
        if isinstance(attr, Module):
            _set_allow_call(attr, value)


def init(model, rng, input):
    """Initializes parameters for a given model and input shape."""
    _set_allow_call(model, True)
    _ = model(input)  # triggers parameter creation
    params = model._collect_params()
    _set_allow_call(model, False)
    return params


def apply(model, params, x):
    """Pure function: apply model with external params."""
    model._assign_params(params)
    _set_allow_call(model, True)
    out = model(x)
    _set_allow_call(model, False)
    return out