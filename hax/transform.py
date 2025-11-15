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

"""transform module for pure functioanal feel"""

import jax
import jax.numpy as jnp
import numpy as np
from .basemodule import Module


def _set_allow_call(module, value: bool):
    """Recursively enable/disable __call__ for a module and all its submodules."""
    module._allow_call = value
    for name, attr in module.__dict__.items():
        if isinstance(attr, Module):
            _set_allow_call(attr, value)


class Transformed:
    """Container for init and apply functions."""
    def __init__(self, init_fn, apply_fn, apply_nonrng):
        self.init = init_fn
        self.apply = apply_fn
        self.apply_nonrng = apply_nonrng
    
    def __repr__(self) -> str:
        return f"TransformedFunction(\ninit:{self.init},\napply:{self.apply},\napply_nonrng:{self.apply})"

from .basemodule import _get_frame

def transform(fun):
    """Return (init_fn, apply_fn) for a pure function `fun` that creates modules inline.

    - init_fn(rng, *args, **kwargs) -> params
    - apply_fn(params, *args, **kwargs) -> outputs
    """

    def init_fn(rng, *args, dtype=None, **kwargs):
        frame = _get_frame()
        # prepare fresh frame for init
        frame.params = {}
        frame.rng = rng
        if dtype == None:
            for arg in args:
                if isinstance(arg, np.ndarray|jax.Array):
                    frame.dtype = arg.dtype
    
        else:
            frame.dtype = dtype
            
        frame.in_init = True
        frame.reset_counters()

        try:
            _ = fun(*args, **kwargs)
        finally:
            # always disable init flag to avoid accidental reuse
            frame.in_init = False
            frame.rng = None
            frame.reset_counters()

        # return the params dict (shallow copy to avoid accidental mutation)
        return frame.params

    def apply_fn(params, rng, *args, **kwargs):
        frame = _get_frame()
        frame.params = params
        frame.in_init = False
        frame.rng = rng 
        frame.reset_counters()

        out = fun(*args, **kwargs)

        frame.params = {}
        return out

    def apply_nonrng(params, *args, **kwargs):
        frame = _get_frame()
        frame.params = params
        frame.in_init = False
        frame.rng = None
        frame.reset_counters()

        out = fun(*args, **kwargs)

        # clear frame.params reference to avoid accidental retention
        frame.params = {}
        return out

    return Transformed(init_fn, apply_fn, apply_nonrng)
