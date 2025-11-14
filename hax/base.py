import jax
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class ModuleTree:
    def __init__(self) -> None:
        pass

    def tree_flatten(self):
        """Flatten the module into its dynamic and static components.

        Returns
        -------
        tuple
            A `(children, aux)` tuple as expected by JAX PyTree utilities.
        """
        children = {
            k: v for k, v in self.__dict__.items()
            if isinstance(v, ModuleTree) or isinstance(v, jax.numpy.ndarray)
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
        ModuleTree
            The reconstructed module.
        """
        cls_type, child_names, static = aux
        obj = cls_type.__new__(cls_type)
        obj.__dict__.update(static)
        for name, value in zip(child_names, children):
            obj.__dict__[name] = value
        return obj