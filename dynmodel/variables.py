if __debug__:
    import types

class Variable(object):
    """
    An object representing a variable in a dynamic state model, with possible
    values in range(minval, maxval) (ie. minval to maxval-1).
    """
    def __init__(self, vals, description="", continuous=False):
        assert type(vals)==int, "the number of values (vals) must be an integer"
        assert type(description) in types.StringTypes, "the description must be a string"
        assert type(continuous)==bool, "must use a boolean to indicate whether this is a continuous variable"

        self.__vals = vals
        self.__descr = description
        self.__continuous = continuous

    def __repr__(self):
        return "%s(%i, '%s', %r)" % (self.__class__, self.__vals,
                self.__descr.replace("'", "\\'"), self.__continuous )

    def discrete(self):
        """
        Is this a discrete variable?
        """
        return not self.__continuous
    
    def continuous(self):
        """
        Is this a continuous variable?
        """
        return self.__continuous
    
    def descr(self):
        """
        The description string for this variable
        """
        return self.__descr
    
    def values(self):
        """
        Return a list of the possible values for this variable.
        """
        return range(self.__vals)
    
    def xvalues(self):
        """
        Return an iterator over possible values for this variable.
        """
        return xrange(self.__vals)
    
    def size(self):
        """
        Return the number of possible (integer) values for this variable.
        """
        return self.__vals
        
    def range(self):
        """
        Return the range of this variable (0, vals).
        """
        return (0, self.__vals)
        


def _isVariable(v):
    return isinstance(v, Variable)
def _conjunction(a, b):
    return a and b

class StateVariables(object):
    """
    An object representing the collection of state variables for a model.
    """
    def __init__(self, variables):
        assert hasattr(variables, '__iter__'), "argument must be a list of variables"
        assert reduce(_conjunction, map(_isVariable, variables)), "not all values in the lise are Variable instances"
        
        self.__vars = list(variables)
    
    def __repr__(self):
        varstrs = [repr(v) for v in self.__vars]
        return "%s([ %s ])" % (self.__class__, ", ".join(varstrs))

    def __allvals(self, vars):
        """
        Recursive algorithm to generate all possible values of the variables
        in the Variable list vars.
        
        Values are generated in row-major order, which is probably a good thing.
        """
        if len(vars)==1:
            for v in vars[0].xvalues():
                yield (v,)
            return
        
        var = vars[0]
        rec = vars[1:]
        for v in var.xvalues():
            for r in self.__allvals(rec):
                yield (v,) + r

    def all(self):
        """
        Return the list of Variable objects represented.
        """
        return self.__vars
        
    def __getitem__(self, ind):
        """
        Return Variable at position ind.
        """
        return self.__vars[ind]

    def len(self):
        """
        Return the number of variables.
        """
        return len(self.__vars)
        
    def shape(self):
        """
        Return the shape of the array implied by these variables (as a list).
        """
        return tuple(v.size() for v in self.__vars)
        
    def allvalues(self):
        """
        Return iterator that gives all possible parameter values.
        """
        for v in self.__allvals(self.__vars):
            yield v

    def ranges(self):
        """
        Return a list of ranges for each variable.
        """
        return [ v.range() for v in self.__vars ]
