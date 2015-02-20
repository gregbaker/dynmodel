import cython

def interpn(qual0, vars, n, t, state):
    """
    Recursively interpolate state variables 0..n.
    """
    return __interpn(qual0, vars, n, t, state)

#@cython.cdivision(True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
cdef double __interpn(qual0, vars, int n, int t, state) except? -2:
    cdef double xc, dx, fit
    cdef int x

    cdef int size

    if n<0:
        # recursive base-case
        return qual0(t, state)

    xc = state[n]
    var = vars[n]

    # check variable bounds
    if xc < 0:
        state[n] = 0
        return __interpn(qual0, vars, n-1, t, state)  # is that okay in general?

    size = var.size()
    if xc >= size - 1:
        state[n] = size - 1
        return __interpn(qual0, vars, n-1, t, state)

    x = int(xc)

    if var.discrete():
        # discrete variable: no need to interpolate
        assert x==xc, "discrete variable given non-integer value: state=" + str(state)
        return __interpn(qual0, vars, n-1, t, state)


    if x==xc:
        # integer: no need to interpolate
        state[n] = x
        return __interpn(qual0, vars, n-1, t, state)

    dx = xc-x

    # manipulate state in-place; recurse; clean up at the end
    state[n] = x
    fit = (1-dx) * __interpn(qual0, vars, n-1, t, state)
    state[n] = x+1
    fit += dx * __interpn(qual0, vars, n-1, t, state)
    state[n] = xc # restore, so backtracks are unaffected

    return fit
