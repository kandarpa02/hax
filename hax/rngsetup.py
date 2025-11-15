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

from .base import ModuleTree
from jax.random import PRNGKey, split
from typing import Iterable


class RNG(ModuleTree):
    """
    Lightweight PRNG wrapper used by Hax for deterministic parameter and
    stochastic operation control.

    This class provides an object-oriented interface around JAX's functional
    `jax.random.PRNGKey`, while preserving *tree friendliness* for use inside
    transformed functions (`init`, `apply`, `jit`, etc.).

    Unlike raw JAX keys, `RNG` supports:
      • Stateful-like `split()` semantics that return new `RNG` objects  
      • Iteration over subkeys  
      • Convenient indexing (`rng[i]`)  
      • Pretty-printing for debugging  
      • `maker()` utility to inject an existing PRNGKey  

    Notes
    -----
    This wrapper does **not** mutate JAX PRNG states internally. It only
    stores a JAX key array and returns split versions of it. All mutations
    (`__setitem__` or overwriting `.key`) are performed on the JAX array,
    not on underlying C++ state.

    Examples
    --------
    Basic usage:
    >>> rng = RNG(42)
    >>> k1 = rng.key
    >>> rng2 = rng.split()          # returns RNG of shape (2, 2)

    Indexing:
    >>> sub = rng2[0]               # RNG keyed by rng2.key[0]

    Iteration:
    >>> for r in rng2:
    ...     print(r.key)

    Passing into transformed functions:
    >>> @hax.transform
    ... def f(x):
    ...     k = hax.RNG(42)
    ...     return x + jax.random.normal(k.key)

    Attributes
    ----------
    seed : int
        Original integer seed used to create the base PRNG key.
    key : jax.Array
        Underlying JAX PRNG key. May be a single key (shape=(2,)) or a
        stack of keys (shape=(N, 2)).
    index : int
        Internal cursor used for iteration.

    """

    def __init__(self, seed: int) -> None:
        """
        Create a new RNG from an integer seed.

        Parameters
        ----------
        seed : int
            Initial seed passed to `jax.random.PRNGKey`.
        """
        super().__init__()
        self.seed = seed
        self.key = PRNGKey(self.seed)
        self.index = 0

    @staticmethod
    def maker(key):
        """
        Construct a new `RNG` wrapper from an existing JAX PRNGKey.

        This method bypasses seeding and directly wraps the given key.

        Parameters
        ----------
        key : jax.Array
            A PRNG key or a batch of PRNG keys.

        Returns
        -------
        RNG
            A new `RNG` instance wrapping the provided key.
        """
        r = RNG(0)
        r.key = key
        return r

    def __repr__(self):
        """
        Return a readable multi-line representation of the RNG key.

        Keys are displayed in a human-friendly stacked layout.
        """
        last = self.key[-1]
        name = lambda k: k.__str__()
        out = f'RNG(['

        for i, k in enumerate(self.key):
            if i != self.key.__len__() - 1:
                if i == 0:
                    out += name(k) + "\n"
                else:
                    out += "     " + name(k) + "\n"
            else:
                out += "     " + name(k)

        out += "])"
        return out

    def split(self, n=2):
        """
        Split the current key into `n` subkeys.

        This matches the behavior of `jax.random.split(key, n)` but returns
        a new `RNG` wrapper.

        Parameters
        ----------
        n : int, default=2
            Number of subkeys to generate.

        Returns
        -------
        RNG
            A new RNG whose `.key` is an array of `n` PRNG subkeys.
        """
        splitted = split(self.key, n)
        return RNG.maker(splitted)

    def __setitem__(self, k, v):
        """
        Replace the PRNG subkey at index `k`.

        This directly assigns into the underlying JAX array.
        """
        self.key[k] = v

    def __getitem__(self, idx):
        """
        Extract a single subkey and wrap it in a new RNG.

        Parameters
        ----------
        idx : int
            Index into the stacked key array.

        Returns
        -------
        RNG
            An RNG object representing `self.key[idx]`.
        """
        return RNG.maker(self.key[idx])

    def __iter__(self):
        """
        Make RNG iterable over its subkeys.

        Returns
        -------
        Iterator[RNG]
        """
        return self

    def __next__(self):
        """
        Return the next subkey during iteration.

        Returns
        -------
        RNG
            A wrapper for the next subkey.

        Raises
        ------
        StopIteration
            When all subkeys are consumed.
        """
        if self.index < len(self.key):
            value = self.key[self.index]
            self.index += 1
            return RNG.maker(value)
        else:
            raise StopIteration

    def __call__(self, *args):
        """
        Return the underlying PRNG key when the object is called.

        Useful inside modules that expect a key-like callable.

        Returns
        -------
        jax.Array
            The underlying PRNG key.
        """
        return self.key

    def __dir__(self) -> Iterable[str]:
        """
        Customize tab-completion and attribute listing.

        Only fields other than 'seed', 'key', and 'index' are shown.
        """
        return [name for name in self.__dict__ if name not in ['seed', 'key', 'index']]
