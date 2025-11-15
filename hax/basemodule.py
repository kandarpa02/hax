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

from .base import ModuleTree
from collections import Counter, defaultdict
import threading
from jax import tree_util
import jax
import jax.numpy as jnp
from jax.random import split
import inspect
from .rngsetup import RNG

class Frame:
    """Runtime frame used during init/apply.


    - params: nested dict of module_name -> {param_name: array}
    - rng: PRNGKey used during init to generate params
    - in_init: bool flag
    - _counters: per-class counters to produce deterministic names in call order
    """
    def __init__(self):
        self.params = {}
        self.rng:RNG = None
        self.dtype = None
        self.in_init = False
        self._counters = defaultdict(int)

    def reset_counters(self):
        self._counters.clear()

    def next_name(self, class_name: str) -> str:
        """Generate a deterministic name for the module call."""
        idx = self._counters[class_name]
        self._counters[class_name] += 1
        return f"{class_name.lower()}" if idx == 0 else f"{class_name.lower()}{idx}"

_thread_local = threading.local()

def _get_frame() -> Frame:
    f = getattr(_thread_local, "frame", None)
    if f is None:
        f = Frame()
        _thread_local.frame = f
    return f

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


class Module(ModuleTree):
    """
    Base class for all neural network modules in Hax.

    Hax uses an ephemeral, Haiku-style module system: module instances act as
    *stateless templates* that define how parameters are created and used inside
    a transformed function. Parameters are never stored inside `Module`
    instances; instead they are stored in a thread-local :class:`Frame`,
    populated during `init()` and read during `apply()`.

    ---------------------------------------------------------------------------
    Core Concepts
    ---------------------------------------------------------------------------
    • **Stateless Modules**
        A `Module` carries only Python configuration (like `units`) but no
        parameters, RNG keys, or state. This ensures that the same module
        instance can be reused freely without side effects.

    • **Deterministic Naming**
        Each module invocation receives a stable, deterministic name derived
        from a per-class counter inside the :class:`Frame`. This ensures that
        parameters created during `init()` are retrieved in the same order
        during `apply()`, even across JIT/pmap/vmap transformations.

    • **Frame-Based Parameter Storage**
        Parameters live inside a nested dictionary:
        
        ``{ module_name -> { param_name -> jax.Array } }``

        The module name is assigned per *call*, not per instance.

    • **Init vs Apply**
        During `init()`:
            - Parameters are created using `init_fn`
            - RNGs are consumed by splitting
        During `apply()`:
            - Parameters are simply retrieved from the Frame

    Subclasses must implement :meth:`call`, which defines the forward pass.

    ---------------------------------------------------------------------------
    Example
    ---------------------------------------------------------------------------
    >>> class Linear(hax.Module):
    ...     def __init__(self, units):
    ...         super().__init__()
    ...         self.units = units
    ...
    ...     def call(self, x):
    ...         w = self.add_params("w", [x.shape[-1], self.units], glorot_uniform)
    ...         b = self.add_params("b", [self.units], zeros)
    ...         return x @ w + b

    >>> @hax.transform
    ... def net(x):
    ...     x = Linear(32)(x)
    ...     return Linear(10)(x)

    >>> params = net.init(hax.RNG(0), x)
    >>> y = net.apply(params, x)
    """

    def __init__(self):
        """
        Initialize a new module instance.

        Modules do not store parameters internally. Any attributes set here
        should be Python-only configuration (e.g., number of units, kernel size,
        activation flags). The dtype for parameter initializers may also be set
        here if needed.
        """
        self.dtype = None

    def _begin_call(self):
        """
        Reset per-call state before a new `__call__` execution.

        This ensures that a module invoked multiple times (e.g. inside a loop,
        sequential stack, or repeated in a functional transform) receives a new
        deterministic name for each invocation.

        Clears the `_call_name` cache so :meth:`_current_call_name` will allocate
        a new name on the first param registration of this call.
        """
        if hasattr(self, "_call_name"):
            del self._call_name

    def _current_call_name(self):
        """
        Return the deterministic name assigned to the current module call.

        The name is created lazily on first access via `Frame.next_name(...)`
        and cached for the rest of the call. Subsequent `add_params` calls
        reuse this name so all parameters for this invocation are grouped into
        the same bucket.

        Returns
        -------
        str
            The module name for this invocation (e.g. "linear", "linear1").
        """
        frame = _get_frame()
        if not hasattr(self, "_call_name"):
            self._call_name = frame.next_name(self.__class__.__name__)
        return self._call_name

    def add_param(self, name: str, shape, init_fn):
        """
        Register or retrieve a parameter belonging to this module call.

        Parameters
        ----------
        name : str
            The local name of the parameter within this module call
            (e.g. "w", "b").
        shape : Sequence[int]
            Shape of the parameter to be created.
        init_fn : Callable
            A parameter initializer. Signature may include:
                (shape), (shape, rng), (shape, dtype), (shape, dtype, rng)  
            The method automatically inspects the initializer’s signature and
            supplies supported keyword arguments.

        Returns
        -------
        jax.Array
            The parameter tensor, either created (during init) or retrieved
            (during apply).

        Notes
        -----
        • During `init()`, a fresh RNG is split and passed to the initializer.  
        • During `apply()`, parameters must already exist; otherwise a
          KeyError is raised.  
        • All parameters from this module call share the same
          `_current_call_name()`.
        """
        frame = _get_frame()

        module_name = self._current_call_name()

        # ensure bucket exists
        if module_name not in frame.params:
            frame.params[module_name] = {}

        bucket = frame.params[module_name]

        # creation path
        if frame.in_init:
            if name not in bucket:
                if frame.rng is None:
                    raise RuntimeError("No RNG available in frame during init")

                key, frame.rng = frame.rng.split(2)
                dtype = frame.dtype

                sig = inspect.signature(init_fn)
                kwargs = {}
                if "shape" in sig.parameters:
                    kwargs["shape"] = tuple(shape)
                if "rng" in sig.parameters or "key" in sig.parameters:
                    kwargs["rng" if "rng" in sig.parameters else "key"] = key
                if "dtype" in sig.parameters:
                    kwargs["dtype"] = dtype

                # Call initializer, falling back to positional args if needed.
                try:
                    param = init_fn(**kwargs)
                except TypeError:
                    args = []
                    if "shape" in sig.parameters:
                        args.append(tuple(shape))
                    if "dtype" in sig.parameters:
                        args.append(dtype)
                    if "rng" in sig.parameters or "key" in sig.parameters:
                        args.append(key)
                    param = init_fn(*args)

                bucket[name] = jnp.asarray(param)

        # retrieval path
        if name not in bucket:
            raise KeyError(f"Parameter {name!r} not found in module {module_name!r}")
        return bucket[name]

    def __call__(self, *args, **kwargs):
        """
        Invoke the module on inputs.

        This method resets call-specific naming state via :meth:`_begin_call`,
        then delegates to :meth:`call`, which subclasses must implement.

        Returns
        -------
        Any
            Output of the module's forward computation.
        """
        self._begin_call()
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """
        Forward pass of the module.

        Subclasses must override this method to implement their computation.
        `call()` is invoked by `__call__()` after preparing call-local state.

        Example
        -------
        >>> class AddOne(hax.Module):
        ...     def call(self, x):
        ...         return x + 1

        Raises
        ------
        NotImplementedError
            Always raised if not overridden.
        """
        raise NotImplementedError
