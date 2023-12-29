import attr
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit, random

if __name__ == "__main__":
    key = random.PRNGKey(7)
    size = 10
    x = random.normal(key, (size,), dtype=jnp.float64)
    x = np.random.randn(size)
    print(x)

    def sq(x: float) -> float:
        return x**2.0

    def si(x: float) -> float:
        return jnp.sin(x)

    @attr.frozen
    class K:
        k: int = 2

    def e(x, data, data2) -> float:
        sum = jnp.array(0.0)
        for i in range(data["k"]):
            sum = sum + x[i]
        return sum

    def f(x) -> float:
        A = 7 * np.eye(len(x))
        return jnp.dot(x, jnp.dot(A, x))

    ge = grad(e)
    print(ge(x, {"k": 3}, ()))
    # gf = jacfwd(grad(f))
    # print(gf(x))
