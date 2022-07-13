import numpy
import numpy as np


def mobius_addition(u, v, s):
    # left addition of u to v with max hyperbolic radius s
    u_norm = np.sqrt((u**2).sum(1))
    v_norm = np.sqrt((v ** 2).sum(1))
    u_dot_v = (u * v).sum(axis=1)
    numerator = (((1 +2/s**2 * u_dot_v + v_norm**2/s**2)*u.T).T + ((1-u_norm**2/s**2)*v.T).T)
    denominator = (1 +2/s**2 * u_dot_v + v_norm**2/s**4*u_norm**2)
    return (numerator.T/denominator).T


def einstein_multiplication(u,r,s):
    u_norm = np.sqrt((u ** 2).sum(1))
    return ((s * np.tanh(r * np.arctanh(u_norm / s)) * u.T)).T / u

