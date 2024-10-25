import os
import random
import string
from functools import wraps
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.host_callback import id_print, id_tap
from tqdm.auto import tqdm
import inspect
from scipy.interpolate import RegularGridInterpolator
from einops import rearrange
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, vmap
import sys


def jqdm(total, argnum=0, decimals=1, **kwargs):
    "Decorate a jax scan body function to include a TQDM progressbar."

    pbar = tqdm(range(100), mininterval=500, **kwargs)

    def _update(cur, transforms):
        amt = float(cur * 100 / total)
        amt = round(amt, decimals)
        if amt != pbar.last_print_n:
            pbar.n = amt
            pbar.last_print_n = amt
            pbar.refresh()

    def update_jqdm(cur):
        id_tap(_update, cur),

    def _jqdm(func):

        @wraps(func)
        def wrapper_body_fun(*args, **kwargs):
            cur = args[argnum]
            update_jqdm(cur)
            result = func(*args, **kwargs)
            return result  # close_tqdm(result, amt)

        return wrapper_body_fun

    return _jqdm


def randkey():
    return jax.random.PRNGKey(random.randint(-1e12, 1e12))


def unique_id(n) -> str:
    """creates unique alphanumeric id w/ low collision probability"""
    chars = string.ascii_letters + string.digits  # 64 choices
    id_str = "".join(random.choice(chars) for _ in range(n))
    return id_str


def epoch_time(decimals=0) -> int:
    return int(time() * (10 ** (decimals)))


def count_params(tree):
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(tree))
    return param_count


def pts_array_from_space(space):
    m_grids = jnp.meshgrid(*space, indexing="ij")
    x_pts = jnp.asarray([m.flatten() for m in m_grids]).T
    return x_pts


def pshape(*args):
    # Get the previous frame in the stack (i.e., the caller's frame)
    frame = inspect.currentframe().f_back
    # Get the caller's local variables
    local_vars = frame.f_locals

    # Build a mapping from id(value) to name(s)
    value_to_names = {}
    for var_name, value in local_vars.items():
        value_id = id(value)
        if value_id in value_to_names:
            value_to_names[value_id].append(var_name)
        else:
            value_to_names[value_id] = [var_name]

    dlim = " | "
    for arg in args:
        value_id = id(arg)
        var_names = value_to_names.get(value_id, ["unknown"])
        # Join multiple variable names if they reference the same object
        var_name_str = ", ".join(var_names)
        if hasattr(arg, "shape"):
            print(f"{var_name_str}: {arg.shape}", end=dlim)
        else:
            print(f"{var_name_str}: no_shape", end=dlim)
    print()


def interplate_in_t(sols, true_t, interp_t):
    sols = np.asarray(sols)
    T, N, D = sols.shape

    data_spacing = [np.linspace(0.0, 1.0, n) for n in sols.shape[1:]]
    spacing = [np.squeeze(true_t), *data_spacing]

    gt_f = RegularGridInterpolator(spacing, sols, method="linear", bounds_error=True)

    interp_spacing = [np.squeeze(interp_t), *data_spacing]
    x_pts = pts_array_from_space(interp_spacing)
    interp_sols = gt_f(x_pts)

    interp_sols = rearrange(interp_sols, "(T N D) -> T N D", N=N, D=D)
    return interp_sols


def get_rand_idx(key, N, bs):
    if bs > N:
        bs = N
    idx = jnp.arange(0, N)
    return jax.random.choice(key, idx, shape=(bs,), replace=False)


def hess_trace_estimator(fn, argnum=0, diff="rev"):

    if diff == "fwd":
        d_fn = jacfwd(fn, argnums=argnum)
    else:
        d_fn = jacrev(fn, argnums=argnum)

    def estimator(key, *args, **kwargs):
        args = list(args)
        primal = args[argnum]
        eps = jax.random.normal(key, shape=primal.shape)

        def s_dx_wrap(x):
            return d_fn(*args[:argnum], x, *args[argnum + 1 :], **kwargs)

        dx_val, jvp_val = jvp(s_dx_wrap, (primal,), (eps,))
        trace = jnp.dot(eps, jvp_val)
        return dx_val, trace

    return estimator


def meanvmap(f, mean_axes=(0,), in_axes=(0,)):
    return lambda *fargs, **fkwargs: jnp.mean(
        vmap(f, in_axes=in_axes)(*fargs, **fkwargs), axis=mean_axes
    )


def tracewrap(f, axis1=0, axis2=1):
    return lambda *fargs, **fkwargs: jnp.trace(
        f(*fargs, **fkwargs), axis1=axis1, axis2=axis2
    )


def batchmap(f, n_batches, argnum=0):

    def wrap(*fargs, **fkwarg):
        fargs = list(fargs)
        X = fargs[argnum]
        batches = jnp.split(X, n_batches, axis=0)

        result = []
        for B in batches:
            fargs[argnum] = B
            a = f(*fargs, **fkwarg)
            result.append(a)

        return jnp.concatenate(result)

    return wrap


def sqwrap(f):
    return lambda *fargs, **fkwargs: jnp.squeeze(f(*fargs, **fkwargs))


def get_lims(tensor, margin=0.0, as_dict=False):

    x_coords = tensor[..., 0]
    y_coords = tensor[..., 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    xlim = (x_min - margin, x_max + margin)
    ylim = (y_min - margin, y_max + margin)

    if as_dict:
        return {"xlim": xlim, "ylim": ylim}

    return xlim, ylim


def count_params(params):
    return sum(x.size for x in jax.tree_leaves(params))


def normalize_data(x, axis=None, method="std"):
    if method == "01":
        mm, mx = x.min(axis=axis, keepdims=True), x.max(axis=axis, keepdims=True)
        shift, scale = mm, (mx - mm)
    else:
        shift, scale = np.mean(x, axis=axis, keepdims=True), np.std(
            x, axis=axis, keepdims=True
        )

    x = (x - shift) / scale

    def unnormalize_fn(data):
        return (data * scale) + shift

    return x, unnormalize_fn


def key_tensor(key, shape):
    num_keys = jnp.prod(jnp.asarray((shape)))
    keys = jax.random.split(
        key, num_keys
    )  # split the key into the required number of keys
    return keys.reshape(shape + (2,))


def fold_in_data(*args):
    s = 0.0
    for a in args:
        s += jnp.cos(jnp.linalg.norm(a))
    s *= 1e6
    s = s.astype(jnp.int32)
    return s
