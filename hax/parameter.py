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


from .basemodule import _get_frame, split, inspect, jnp
from .basemodule import Module
from typing import Callable

def param(module:Module, name:str, shape:tuple|list, init_fn:Callable):
    """
    Register or retrieve a parameter belonging to this module call.

    Parameters
    ----------
    module: Module
        Module instance which is the current subclass you're working on
        (e.g. "self").
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
    return module.add_param(name, shape, init_fn)
