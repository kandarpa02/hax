from .basemodule import _get_frame, split, inspect, jnp

def Param(module, name, shape, init_fn):
    """
    Standalone Haiku-like get_parameter function.
    
    Parameters
    ----------
    module : Module
        The module instance calling this parameter.
    name : str
        Name of the parameter.
    shape : tuple or list
        Shape of the parameter.
    init_fn : callable
        Initialization function that takes `shape` and optionally `rng` or `key`.

    Returns
    -------
    jnp.ndarray
        The parameter array from the frame.
    """
    frame = _get_frame()
    module_name = module._module_name()  # call module's name function

    # ensure module bucket exists
    if module_name not in frame.params:
        frame.params[module_name] = {}

    bucket = frame.params[module_name]

    if frame.in_init:
        # create param if not present
        if name not in bucket:
            # split rng for param creation
            if frame.rng is None:
                raise RuntimeError("No RNG available in frame during init")
            key, frame.rng = split(frame.rng, 2)
            dtype = frame.dtype

            # call init_fn with supported kwargs
            sig = inspect.signature(init_fn)
            kwargs = {}
            if "shape" in sig.parameters:
                kwargs["shape"] = tuple(shape)
            if "rng" in sig.parameters or "key" in sig.parameters:
                # prefer rng kwarg name if present
                if "rng" in sig.parameters:
                    kwargs["rng"] = key
                else:
                    kwargs["key"] = key
                    
            if "dtype" in sig.parameters:
                kwargs["dtype"] = dtype

            # support functions that expect positional-only (shape, rng)
            try:
                param = init_fn(**kwargs)
            except TypeError:
                # fallback to positional
                pos_args = []
                if "shape" in sig.parameters:
                    pos_args.append(tuple(shape))
                if "dtype" in sig.parameters:
                    pos_args.append(dtype)
                if ("rng" in sig.parameters) or ("key" in sig.parameters):
                    pos_args.append(key)
                param = init_fn(*pos_args)

            bucket[name] = jnp.asarray(param)

    if name not in bucket:
        raise KeyError(f"Parameter {name} not found in module {module_name}")
    
    return bucket[name]
