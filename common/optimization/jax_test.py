import attr
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit, random, hessian

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

    def e(x, data) -> float:
        sum = jnp.array(0.0)
        for i in range(data.k):
            sum = sum + x[i]
        return sum

    def ee(x) -> float:
        A = 2 * np.eye(len(x))
        return A @ x

    def f(x) -> float:
        A = 7 * np.eye(len(x))
        return jnp.dot(x, jnp.dot(A, x))

    # @jax.tree_util.register_pytree_node_class
    @attr.frozen
    class PK:
        k: int = 2

        # @classmethod
        # def tree_unflatten(cls, aux_data, children):
        #    return cls(*children)

    def ff(x, p: PK) -> float:
        A = p.k * np.eye(len(x))
        return jnp.dot(x, jnp.dot(A, x))

    print(grad(sq)(3.0), jacfwd(sq)(3.0), hessian(sq)(3.0))

    ge = grad(e)
    print(ge(x, K(3)), jacfwd(e)(x, K(3)))

    # gee = jacfwd(ee)
    # print(gee(x))
    # print(hessian(ee)(x))

    # gf = jacfwd(grad(f))
    # print(gf(x))
    # print(hessian(f)(x))

    p = PK(2)
    print(type(p))
    gff = jacfwd(grad(ff))
    print(gff(x, p))
