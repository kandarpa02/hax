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


class Frame:
    """Runtime frame used during init/apply.


    - params: nested dict of module_name -> {param_name: array}
    - rng: PRNGKey used during init to generate params
    - in_init: bool flag
    - _counters: per-class counters to produce deterministic names in call order
    """
    def __init__(self):
        self.params = {}
        self.rng = None
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
    """Base class for ephemeral modules in Haiku-style Hax.

    Modules are *templates* only. They do not persist parameters. On each
    call during init/apply they generate deterministic names via the Frame and
    read/write params from the Frame.params dict.
    """

    def __init__(self):
        # store any python-only configuration in the instance (e.g., units)
        pass

    def _module_name(self) -> str:
        """Return the deterministic module name for the current call.

        This uses the Frame.next_name() which increments counters in order of
        creation. The same creation order in init and apply yields identical
        names.
        """
        frame = _get_frame()
        return frame.next_name(self.__class__.__name__)


    def add_params(self, name: str, shape, init_fn):
        """Register or fetch a parameter for this module call.

        Behavior depends on whether we're inside init or apply:
        - init: call `init_fn` to create the parameter, store it into
          `frame.params[module_name][name]`.
        - apply: read value from `frame.params` and return it.

        `init_fn` may accept kwargs like (shape=..., rng=...) or (shape,)
        and we introspect the signature.
        """
        frame = _get_frame()
        module_name = self._module_name()

        # ensure module bucket exists in params
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

                # support functions that expect positional-only (shape, rng)
                try:
                    param = init_fn(**kwargs)
                except TypeError:
                    # fallback to positional
                    pos_args = []
                    if "shape" in sig.parameters:
                        pos_args.append(tuple(shape))
                    if ("rng" in sig.parameters) or ("key" in sig.parameters):
                        pos_args.append(key)
                    param = init_fn(*pos_args)

                bucket[name] = jnp.asarray(param)

        # apply or post-init: return the parameter
        if name not in bucket:
            raise KeyError(f"Parameter {name} not found in module {module_name}")
        return bucket[name]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

