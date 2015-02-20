"""
Pure-Python implementations of everything in criticalx.pyx, so we can fall back.
"""

def index(t, state):
    return (t,) + tuple(state)

def qual0(qual, t, state):
    return qual[index(t, state)]

def interpn(qual, vars, n, t, state):
    """
    Recursively interpolate state variables 0..n.
    """
    if n<0:
        # recursive base-case
        return qual0(qual, t, state)

    xc = state[n]
    var = vars[n]

    # check variable bounds
    if xc < 0:
        state[n] = 0
        return interpn(qual, vars, n-1, t, state)
    if xc >= var.size() - 1:
        state[n] = var.size() - 1
        return interpn(qual, vars, n-1, t, state)

    x = int(xc)

    if var.discrete():
        # discrete variable: no need to interpolate
        assert x==xc, "discrete variable given non-integer value: state=" + str(state)
        return interpn(qual, vars, n-1, t, state)

    if x==xc:
        # integer: no need to interpolate
        state[n] = x
        return interpn(qual, vars, n-1, t, state)

    dx = xc-x

    # manipulate state in-place; recurse; clean up at the end
    state[n] = x
    fit = (1-dx) * interpn(qual, vars, n-1, t, state)
    state[n] = x+1
    fit += dx * interpn(qual, vars, n-1, t, state)
    state[n] = xc # restore, so backtracks are unaffected

    return fit
