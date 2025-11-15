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


from typing import Sequence, Callable, Union
from .basemodule import Module

class Sequential(Module):
    """
    A container module that applies a sequence of layers or callables in order.

    `Sequential` allows you to chain multiple modules or functions
    together in a simple, functional style, similar to `torch.nn.Sequential`
    or `haiku.Sequential`.

    Example
    -------
    >>> seq = Sequential([
    ...     Linear(128),
    ...     jax.nn.relu,
    ...     Linear(10)
    ... ])
    
    Parameters
    ----------
    layers : Sequence[Module | Callable]
        A sequence of modules (inheriting from `Module`) or plain callables
        (like activation functions) to apply in order.

    Notes
    -----
    - Supports any callable that takes a single argument (the input) and
      returns the output.
    - Works seamlessly with Hax's ephemeral modules and inline parameter
      management.
    - Useful for creating MLPs or simple layer stacks with minimal boilerplate.
    """

    def __init__(self, layers: Sequence[Union[Module, Callable]]):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
