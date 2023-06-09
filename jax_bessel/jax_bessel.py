import jax.numpy as jnp
import scipy.special
from jax import jit
from jax import custom_jvp, pure_callback, vmap

def _generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: -cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) - cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv


def _generate_modified_bessel(function, sign):
    """function is Kv and Iv"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: sign * cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) + cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv

def _spherical_bessel_genearator(f):
    def g(v, x):
        return f(v + 0.5, x) * jnp.sqrt(jnp.pi / (2 * x))

    return g

jv = _generate_bessel(scipy.special.jv)
yv = _generate_bessel(scipy.special.yv)
hankel1 = _generate_bessel(scipy.special.hankel1)
hankel2 = _generate_bessel(scipy.special.hankel2)
kv = _generate_modified_bessel(scipy.special.kv, sign=-1)
iv = _generate_modified_bessel(scipy.special.iv, sign=+1)
spherical_jv = _spherical_bessel_genearator(jv)
spherical_yv = _spherical_bessel_genearator(yv)
spherical_hankel1 = _spherical_bessel_genearator(hankel1)
spherical_hankel2 = _spherical_bessel_genearator(hankel2)