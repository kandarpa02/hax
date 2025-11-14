from .base import ModuleTree
from jax.random import PRNGKey, split
from typing import Iterable

class RNG(ModuleTree):
    def __init__(self, seed:int) -> None:
        super().__init__()
        self.seed = seed
        self.key = PRNGKey(self.seed)
        self.index=0

    @staticmethod
    def maker(key):
        r = RNG(0)
        r.key = key
        return r
            
    def __repr__(self):
        last = self.key[-1]
        name = lambda k:k.__str__()
        out = f'RNG(['
        # out += '    '.join(f'{name(k)}\n' for k in self.key if k!=last)

        for i, k in enumerate(self.key):
            if i != self.key.__len__() - 1:
                if i == 0:
                    out += name(k) + '\n'
                else:
                    out += '     ' + name(k) + '\n'
            else:
                out += '     ' + name(k)

        out += '])'
        return out
    
    # __str__ = __repr__

    def split(self, n=2):
        splitted = split(self.key, n)
        return RNG.maker(splitted)
    
    def __setitem__(self, k, v):
        self.key[k] = v

    def __getitem__(self, idx):
        return RNG.maker(self.key[idx])
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < len(self.key):
            value = self.key[self.index]
            self.index += 1
            return RNG.maker(value)
        else:
            raise StopIteration

    def __call__(self, *args):
        return self.key
    
    def __dir__(self) -> Iterable[str]:
        return [name for name in self.__dict__ if not name in ['seed', 'key', 'index']]