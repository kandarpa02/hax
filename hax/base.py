from collections import Counter
from jax import tree_util
import jax
from jax.random import split

def _set_rng(module, rng):
    module.rng, _ = split(rng, 2)
    for name, attr in module.__dict__.items():
        if isinstance(attr, Module):
            rng, _ = jax.random.split(rng, 2)
            _set_rng(attr, rng)

@tree_util.register_pytree_node_class
class Module:
    def __init__(self) -> None:
        self._params = {}
        self._allow_call = False
        self.rng = None
        self.dtype = None

    def _set_rng(self):
        _set_rng(self, self.rng)

    def _set_dtype(self, dtype):
        self.dtype = dtype
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                attr.dtype = dtype

    def add_parameters(self, name, shape, init_function):
        dtype = self.dtype
        try:
            param = init_function(shape=shape, dtype=dtype, rng=self.rng)
        except:
            param = init_function(shape=shape, dtype=dtype)

        self._params[name] = jax.numpy.asarray(param, dtype=dtype)
        return self._params[name]

    def _collect_params(self):
        params = dict(self._params)
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                params[name] = attr._collect_params()
        return params

    def _assign_params(self, params):
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                attr._assign_params(params[name])
        self._params = {k: v for k, v in params.items() if not isinstance(v, dict)}

    def tree_flatten(self):
        children = {k: v for k, v in self.__dict__.items() if isinstance(v, Module) or isinstance(v, jax.numpy.ndarray)}
        static = {k: v for k, v in self.__dict__.items() if k not in children}
        return (tuple(children.values()), (type(self), tuple(children.keys()), static))

    @classmethod
    def tree_unflatten(cls, aux, children):
        cls_type, child_names, static = aux
        obj = cls_type.__new__(cls_type)
        obj.__dict__.update(static)
        for name, value in zip(child_names, children):
            obj.__dict__[name] = value
        return obj

    def __call__(self, *args, **kwargs):
        if not getattr(self, "_allow_call", False):
            raise RuntimeError("Direct model call is not allowed. Use `apply(model, params, x)` instead.")
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
