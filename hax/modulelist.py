from .basemodule import Module
from typing import Protocol, runtime_checkable, Union, Callable, Type

@runtime_checkable
class LayerFactory(Protocol):
    def __call__(self) -> Module:
        ...

LayerSpec = Union[
    Module,
    Type[Module],
    LayerFactory,
]


class ModuleStack(Module):

    """A container that holds a sequential stack of Modules.

    `ModuleStack` replicates a given layer specification multiple times,
    similar to Haiku's `Sequential` or PyTorch's `ModuleList`. Each
    submodule is registered so its parameters participate in Hax's
    parameter collection and assignment mechanisms.

    Args:
        layer (Module | type[Module] | Callable):
            The layer specification to replicate. Accepts:
            - An instance of a Module
            - A Module subclass (instantiated with `**kwargs`)
            - A factory function returning a Module
        nums (int):
            Number of layers to create.
        **kwargs:
            Keyword arguments passed to the layer constructor when
            `layer` is a class.

    Notes:
        This container does not implement `__call__`. It simply provides
        iteration and indexing. Computation must be defined in the parent
        module.
    """

    def __init__(self, layer:LayerSpec, nums:int, **kwargs) -> None:
        super().__init__()
            
        self.nums = nums
        self.layers = self._make_stack(layer, **kwargs)
    
    def _make_stack(self, layer, **kwargs):
        layers = []
        for i in range(self.nums):
            if isinstance(layer, Module):
                obj = layer
            elif isinstance(layer, type) and issubclass(layer, Module):
                obj = layer(**kwargs)                            
            elif callable(layer):
                obj = layer()                      
            else:
                raise TypeError("Layer must be a Module, class, or factory.")

            name = f"{obj.__class__.__name__.lower()}{i}"
            setattr(self, name, obj)
            layers.append(obj)

        return layers
    

    def __setitem__(self, k, v):
        """Assign a module at index `k`.

        Args:
            k (int): Index to set.
            v (Module): Module to assign.
        """
        self.layers[k] = v

    def __getitem__(self, idx):
        """Retrieve a module or slice of modules.

        Args:
            idx (int | slice): Position or slice.

        Returns:
            Module | list[Module]: Selected module(s).
        """
        return self.layers[idx]

    def __iter__(self):
        """Iterate over submodules.

        Returns:
            iterator: Iterator over layers.
        """
        return iter(self.layers)

    def __len__(self):
        """Number of layers in the stack.

        Returns:
            int: Length of the layer list.
        """
        return len(self.layers)