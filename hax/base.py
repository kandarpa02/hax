# Copyright 2025 Kandarpa Sarkar.
#
# Licensed under the MIT License.
# You may obtain a copy of the License at:
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Hax: A lightweight Module abstraction for JAX.

"""Module Class"""

from collections import Counter
from jax import tree_util
import jax
from jax.random import split


def _set_allow_call(module, value: bool):
    """Recursively enable or disable the `__call__` method for a module tree.

    Parameters
    ----------
    module : Module
        The root module whose call permission is to be updated.
    value : bool
        Whether to allow (`True`) or disallow (`False`) direct calls.
    """
    module._allow_call = value
    for name, attr in module.__dict__.items():
        if isinstance(attr, Module):
            _set_allow_call(attr, value)


def _set_rng(module, rng):
    """Recursively assign RNG keys to a module and its submodules.

    Parameters
    ----------
    module : Module
        The root module to assign random number generator keys to.
    rng : jax.random.PRNGKey
        The RNG key to split and distribute across the module tree.
    """
    module.rng, _ = split(rng, 2)
    for name, attr in module.__dict__.items():
        if isinstance(attr, Module):
            rng, _ = jax.random.split(rng, 2)
            _set_rng(attr, rng)


@tree_util.register_pytree_node_class
class Module:
    """Base class for all neural network modules in Hax.

    The `Module` abstraction provides a low-level interface for defining
    parameterized layers in JAX. It supports parameter management, RNG
    propagation, and JAX tree utilities for functional transformations.
    """

    def __init__(self) -> None:
        """Initialize an empty module."""
        self._params = {}
        self._allow_call = False
        self.rng = None
        self.dtype = None

    def _set_rng(self):
        """Assign RNG keys to this module and all submodules."""
        _set_rng(self, self.rng)

    def _set_dtype(self, dtype):
        """Propagate the dtype setting to this module and all submodules.

        Parameters
        ----------
        dtype : jax.numpy.dtype
            The data type to apply to parameters and submodules.
        """
        self.dtype = dtype
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                attr.dtype = dtype

    def add_params(self, name, shape, init_function):
        """Register a learnable parameter for the module.

        Parameters
        ----------
        name : str
            The name of the parameter.
        shape : tuple[int]
            The shape of the parameter.
        init_function : Callable
            A function to initialize the parameter. Must accept
            `(shape, dtype, rng)` or `(shape, dtype)`.

        Returns
        -------
        jax.numpy.ndarray
            The initialized parameter array.
        """
        dtype = self.dtype
        try:
            param = init_function(shape=shape, dtype=dtype, rng=self.rng)
        except Exception:
            param = init_function(shape=shape, dtype=dtype)

        self._params[name] = jax.numpy.asarray(param, dtype=dtype)
        return self._params[name]
    
    def get_params(self, name):
        return self._params[name]

    def _collect_params(self):
        """Recursively gather parameters from all submodules.

        Returns
        -------
        dict
            A nested dictionary containing all module parameters.
        """
        params = dict(self._params)
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                params[name] = attr._collect_params()
        return params

    def _assign_params(self, params):
        """Recursively assign parameters to this module and its submodules.

        Parameters
        ----------
        params : dict
            A nested dictionary of parameters, typically from `_collect_params()`.
        """
        for name, attr in self.__dict__.items():
            if isinstance(attr, Module):
                attr._assign_params(params[name])
        self._params = {k: v for k, v in params.items() if not isinstance(v, dict)}

    def tree_flatten(self):
        """Flatten the module into its dynamic and static components.

        Returns
        -------
        tuple
            A `(children, aux)` tuple as expected by JAX PyTree utilities.
        """
        children = {
            k: v for k, v in self.__dict__.items()
            if isinstance(v, Module) or isinstance(v, jax.numpy.ndarray)
        }
        static = {k: v for k, v in self.__dict__.items() if k not in children}
        return (tuple(children.values()), (type(self), tuple(children.keys()), static))

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Reconstruct a module from flattened data.

        Parameters
        ----------
        aux : tuple
            Auxiliary data returned from `tree_flatten`.
        children : tuple
            Flattened child values.

        Returns
        -------
        Module
            The reconstructed module.
        """
        cls_type, child_names, static = aux
        obj = cls_type.__new__(cls_type)
        obj.__dict__.update(static)
        for name, value in zip(child_names, children):
            obj.__dict__[name] = value
        return obj

    def __call__(self, *args, **kwargs):
        """Invoke the module's forward computation.

        Raises
        ------
        RuntimeError
            If direct calls are not allowed (must use `apply()` instead).
        """
        if not getattr(self, "_allow_call", False):
            raise RuntimeError(
                "Direct model call is not allowed. "
                "Use `apply(model, params, x)` instead."
            )
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Define the forward computation for the module.

        Subclasses must override this method to specify computation.

        Raises
        ------
        NotImplementedError
            Always raised in the base class.
        """
        raise NotImplementedError

    def init(self, rng, *args, **kwargs):
        """Initialize module parameters using an example input.

        This method triggers parameter creation by temporarily allowing
        a forward call. All created parameters are collected into a
        nested dictionary.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            RNG key used for parameter initialization.
        *args, **kwargs
            Example inputs passed to `forward()`.

        Returns
        -------
        dict
            A nested dictionary of initialized parameters.
        """
        def init_self(rng, *args, **kwargs):
            if not isinstance(self, Module):
                raise TypeError("The function must be a Module instance.")
            _set_allow_call(self, True)
            self.rng = rng
            self._set_rng()

            dtype = None
            for k, v in kwargs.items():
                if k in ['dtype']:
                    dtype = v
                else:
                    dtype = args[0].dtype

            self._set_dtype(dtype=dtype)
            _ = self(*args)  # trigger parameter creation
            _set_allow_call(self, False)
            params = self._collect_params()
            return params

        return init_self(rng, *args, **kwargs)

    def apply(self, rng, params, *args):
        """Apply the module with a given set of parameters.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            RNG key for stochastic operations.
        params : dict
            The parameter dictionary obtained from `init()`.
        *args
            Inputs to the forward computation.

        Returns
        -------
        Any
            The module output for the given parameters and inputs.
        """
        def apply_self(rng, params, *args):
            self._assign_params(params)
            _set_allow_call(self, True)
            self.rng = rng
            self._set_rng()
            out = self(*args)
            _set_allow_call(self, False)
            return out

        return apply_self(rng, params, *args)
