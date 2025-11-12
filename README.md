# Hax
*Minimalist Neural Network Abstraction for JAX*

**Hax** is an extremely low-level Module abstraction for building neural networks with JAX. Designed for researchers and enthusiasts who want full control without the overhead of complex frameworks like Haiku or Flax. Build layers from scratch with a learning curve that's practically zero!

## Features

- **Minimal Abstraction**: Just enough structure to organize your code, not enough to get in your way
- **Full JAX Compatibility**: Seamless integration with `jax.jit`, `jax.grad`, and the entire JAX ecosystem
- **Parameter Management**: Simple yet powerful parameter creation and tracking
- **Research-Friendly**: Perfect for experimental architectures and cutting-edge research
- **Zero Boilerplate**: Focus on your model, not framework specifics
---

## Simple Example:

**Model Definition**

```python
import hax
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, uniform

class SimpleMLP(hax.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
    
    def forward(self, x):
        # Custom parameter initialization
        def make_weights(layer_idx, in_dim, out_dim):
            w_init = lambda shape, rng, dtype: uniform(key=rng, shape=shape, dtype=dtype)
            b_init = lambda shape, dtype: jnp.zeros(shape, dtype=dtype)
            
            w = self.add_parameter(f"w{layer_idx}", shape=(in_dim, out_dim), init_function=w_init)
            b = self.add_parameter(f"b{layer_idx}", shape=(out_dim,), init_function=b_init)
            return w, b
        
        # Build the network
        for i, units in enumerate(self.sizes):
            in_dim = x.shape[-1]
            w, b = make_weights(i, in_dim, units)
            
            x = jnp.matmul(x, w) + b
            if i != len(self.sizes) - 1:  # No activation after last layer
                x = jax.nn.relu(x)
        
        return x
```

**Parameter creation and inferene with JAX JIT support**

```python
from hax import transform

model = SimpleMLP([5, 2, 1])
x = jnp.ones([5, 5])

f = transform(model) 
print(f)
# returns init and apply 
# TransformedFunction(
# init:<function transform.<locals>.init_fn at 0x7ad4a178f4c0>,
# apply:<function transform.<locals>.apply_fn at 0x7ad4a01cd9e0>
# )

params = f.init(key, x) # rng key is mandatory for both init and apply
print(params)
# {'w0': Array([[0.7445246 , 0.24972165, 0.08126879, 0.25234675, 0.25021398],
#         [0.2971009 , 0.65872633, 0.26449287, 0.24253619, 0.62358606],
#         [0.7671927 , 0.69710934, 0.49010003, 0.56120205, 0.5469024 ],
#         [0.77456427, 0.13981938, 0.2875079 , 0.4017681 , 0.00849044],
#         [0.62404764, 0.870381  , 0.25616217, 0.16287422, 0.73356557]],      dtype=float32),
#  'b0': Array([0., 0., 0., 0., 0.], dtype=float32),
#  'w1': Array([[0.7445246 , 0.24972165],
#         [0.08126879, 0.25234675],
#         [0.25021398, 0.2971009 ],
#         [0.65872633, 0.26449287],
#         [0.24253619, 0.62358606]], dtype=float32),
#  'b1': Array([0., 0.], dtype=float32),
#  'w2': Array([[0.7445246 ],
#         [0.24972165]], dtype=float32),
#  'b2': Array([0.], dtype=float32)}

jax.jit(f.apply)(key, params, x)
# Array([[4.2896457],
#        [4.2896457],
#        [4.2896457],
#        [4.2896457],
#        [4.2896457]], dtype=float32)
```
---

If you liked my work do star this repo :)
  