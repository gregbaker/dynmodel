"""
The critical inner-loops that are worth doing in Cython to speed things up.
"""

import cython

def index(t, state):
    return (t,) + tuple(state)

def qual0(qual, t, state):
    return qual[index(t, state)]

def interpn(qual, vars, n, t, state):
    """
    Recursively interpolate state variables 0..n.
    """
    return __interpn(qual, vars, n, t, state)

cdef double __interpn(qual, vars, int n, int t, state) except? -2:
    cdef double xc, dx, fit
    cdef int x

    cdef int size

    if n<0:
        # recursive base-case
        return qual0(qual, t, state)

    xc = state[n]
    var = vars[n]

    # check variable bounds
    if xc < 0:
        state[n] = 0
        return __interpn(qual, vars, n-1, t, state)

    size = var.size()
    if xc >= size - 1:
        state[n] = size - 1
        return __interpn(qual, vars, n-1, t, state)

    x = int(xc)

    if var.discrete():
        # discrete variable: no need to interpolate
        assert x == xc
        return __interpn(qual, vars, n-1, t, state)

    if x==xc:
        # integer: no need to interpolate
        state[n] = x
        return __interpn(qual, vars, n-1, t, state)

    dx = xc-x

    # manipulate state in-place; recurse; clean up at the end
    state[n] = x
    fit = (1-dx) * __interpn(qual, vars, n-1, t, state)
    state[n] = x+1
    fit += dx * __interpn(qual, vars, n-1, t, state)
    state[n] = xc # restore, so backtracks are unaffected

    return fit