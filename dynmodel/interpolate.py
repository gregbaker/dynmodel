def interpn(qual0, vars, n, t, state):
    """
    Recursively interpolate state variables 0..n.
    """
    if n<0:
        # recursive base-case
        return qual0(t, state)

    xc = state[n]
    var = vars[n]

    # check variable bounds
    if xc < 0:
        state[n] = 0
        return interpn(qual0, vars, n-1, t, state)  # is that okay in general?
    if xc >= var.size() - 1:
        state[n] = var.size() - 1
        return interpn(qual0, vars, n-1, t, state)

    if var.discrete():
        # discrete variable: no need to interpolate
        assert int(xc)==xc, "discrete variable given non-integer value: state=" + str(state)
        return interpn(qual0, vars, n-1, t, state)

    x = int(xc)
    if x==xc:
        # integer: no need to interpolate
        state[n] = x
        return interpn(qual0, vars, n-1, t, state)

    dx = xc-x

    # manipulate state in-place; recurse; clean up at the end
    state[n] = x
    fit = (1-dx) * interpn(qual0, vars, n-1, t, state)
    state[n] = x+1
    fit += dx * interpn(qual0, vars, n-1, t, state)
    state[n] = xc # restore, so backtracks are unaffected

    return fit
