import sys
import random
from variables import Variable, StateVariables

try:
    from interpolatex import interpn
except ImportError:
    sys.stderr.write("Can't import Cython implementation. Falling back to slower pure-Python.\n")
    from interpolate import interpn

if __debug__:
    import types
    import operator

from numpy import zeros, memmap, fromfile as ff
def fromfile(fh, type, dim):
    arr = ff(fh, dtype=type)
    arr.reshape(*dim)
    return arr

qual_type = 'float'
all = slice(None)


class Model(object):
    """
    An object representing a dynamic model with the time variable ranging from
    0 to tmax-1, and state determined by the variables in the list vars.
    """

    #################################################################
    # Constructor

    def __init__(self, tmax, variables, terminal_quality, possible_outcomes,
            qual_filename='quality.data', dec_filename='decisions.data',
            dead_fitness=0.0, impossible_fitness=-1.0,
            dec_type='ubyte'):
        """
        The possible_outcomes function should return an Outcomes object
        representing possible decisions and corresponding outcomes with
        their probabilities.
        """
        assert type(tmax)==int
        assert callable(terminal_quality) or terminal_quality==None
        assert callable(possible_outcomes) or possible_outcomes==None
        assert operator.isNumberType(dead_fitness)
        assert operator.isNumberType(impossible_fitness)

        self.tmax = tmax
        self.vars = StateVariables(variables)
        self.terminal = terminal_quality
        self.possibilities = possible_outcomes
        self.deadfitness = dead_fitness
        self.impossiblefitness = impossible_fitness
        self.dec_type = dec_type

        self.qual_filename = qual_filename
        self.dec_filename = dec_filename

        # create our arrays
        # actual memory allocation is deferred until actually needed.
        self.__qual = None
        self.__dec = None
        self.__distrib = None
        self.__filleddec = False
        self.__filledqual = False
        self.__filledterminal = False
        
        # these are used many times, so they're calculated once here
        # okay, maybe that's silly, but I haven't fixed it yet
        self.__allstates = list(self.vars.allvalues())
        self.__lenm1 = self.vars.len()-1
        self.__len = self.vars.len()
            
    
    #################################################################
    # Array Creation Functions

    def __qualarraydim(self):
        """
        Return the dimension list for our quality array.
        """
        return (self.tmax,) + self.vars.shape()

    def __decarraydim(self):
        """
        Return the dimension list for our decision array.
        """
        # no decisions are made at time tmax, so it's not needed.
        # plus it lets us get out-of-bounds errors if we make the
        # mistake somewhere else.
        return (self.tmax-1,) + self.vars.shape()
    
    def __create_qualarray(self):
        """
        Allocate the quality array for this model, if not already done.
        """
        if self.__qual != None:
            return

        self.__qual = memmap(filename=self.qual_filename, mode='w+', shape=self.__qualarraydim(), dtype=qual_type)
        self.__filledqual = False
        self.__filledterminal = False

    def __create_decarray(self):
        """
        Allocate the decision array for this model, if not already done.
        """
        if self.__dec != None:
            return

        self.__dec = memmap(filename=self.dec_filename, mode='w+', shape=self.__decarraydim(), dtype=self.dec_type)
        self.__filleddec = False

    def __create_distribarray(self):
        """
        Allocate the distribution array for this model.
        """
        # this is allowed to recreate so it fills with zeros, thus initializing properly
        self.__distrib = zeros(shape=self.__qualarraydim(), type="Float")

    
    
    #################################################################
    # General/Utility functions

    def __repr__(self):
        return "%s(%r, %r, ...)" % (self.__class__, self.tmax, self.vars)

    def index(self, t, state):
        """
        Convert from a time and state value to an array index.
        
        Identity function for the moment.
        """
        assert hasattr(state, '__iter__') # isiterable
        assert len(state) == self.__len
        assert type(t) == int
        assert t <= self.tmax

        #ind = [x-var.minval for x,var in zip(state, self.vars.all())]
        return (t,) + tuple(state)
        
    def quality(self, t, *state):
        """
        Return the quality value at the given time and state.
        """
        return self.__qual[self.index(t, state)]

    def qual0(self, t, state):
        """
        Return the quality value at the given time and state.

        This function assumes that the state is passed as a list/tuple of
        values.  It is intended for internal use, but could be used outside
        as well.
        """
        return self.__qual[self.index(t, state)]

    def decision(self, t, *state):
        """
        Return the decision value at the given time and state.
        """
        return self.__dec[self.index(t, state)]

    def decision0(self, t, state):
        """
        Return the decision value at the given time and state.
        
        This function assumes that the state is passed as a list/tuple of
        values.  It is intended for internal use, but could be used outside
        as well.
        """
        return self.__dec[self.index(t, state)]
    
    #################################################################
    # Linear Interpolation

    def __interpn(self, n, t, state):
        """
        Recursively interpolate state variables 0..n.
        """
        # If anybody ever gets the idea to speed something up by coding in
        # C or Pyrex, do this one.  It's typically half the total running time.
        if n<0:
            # recursive base-case
            return self.qual0(t, state)
            
        xc = state[n]
        var = self.vars[n]

        # check variable bounds
        if xc < 0:
            state[n] = 0
            return self.__interpn(n-1, t, state)  # is that okay in general?
        if xc >= var.size() - 1:
            state[n] = var.size() - 1
            return self.__interpn(n-1, t, state)

        if var.discrete():
            # discrete variable: no need to interpolate
            assert int(xc)==xc, "discrete variable given non-integer value: state=" + str(state)
            return self.__interpn(n-1, t, state)
        
        x = int(xc)
        if x==xc:
            # integer: no need to interpolate
            state[n] = x
            return self.__interpn(n-1, t, state)

        dx = xc-x
        
        # manipulate state in-place; recurse; clean up at the end
        state[n] = x
        fit = (1-dx) * self.__interpn(n-1, t, state)
        state[n] = x+1
        fit += dx * self.__interpn(n-1, t, state)
        state[n] = xc # restore, so backtracks are unaffected
        
        return fit
        
    def interp(self, t, *state):
        """
        Interpolate to (possibly) fractional values of the state
        variables, from neighbouring integer values.
        """
        assert hasattr(state, '__iter__') # isiterable
        assert len(state) == self.__len        

        state = list(state) # needs to be mutable for the recursive algorithm
        #return self.__interpn(self.__lenm1, t, state)
        return interpn(self.qual0, self.vars, self.__lenm1, t, state)

    
    #################################################################
    # Dynamic Programming Functions

    def fill_terminal_quality(self):
        """
        Set the quality at t=tmax-1 from the model's terminal quality function.
        """
        assert self.terminal != None
        
        self.__create_qualarray()
        for state in self.vars.allvalues():
            self.__qual[ self.index(self.tmax-1, state) ] = self.terminal(*state)
        
        self.__filledterminal = True

    def fill_state(self, t, state, choicefunc=max):
        """
        Fill in quality and decision for one state+time.
        """
        index = self.index(t, state)

        # get all expected values for this state
        outcomes = self.possibilities(self, t, *state)
        expected = outcomes.all_expected()

        # choose according to the choice function

        q, d = choicefunc((v,k) for (k,v) in expected.iteritems())
        # (that list comprehension maps {k1:v1, k2:v2} to [(v1,k1), (v2,k2)]
        # ie. a list of (value, key) pairs)
        return index, q, d

    def fill_time(self, t, choicefunc=max):
        """
        Do a single dynamic programming step: fill in the quality and decision
        matrices at time t.
        """
        assert type(t)==types.IntType and 0<=t<=self.tmax, "t must be an integer from 0 to tmax"
        assert callable(choicefunc), "choicefunc must be a function"

        for state in self.__allstates:
            index, q, d = self.fill_state(t, state, choicefunc)
            (self.__qual[index], self.__dec[index]) = q, d


    def fill_quality(self, choicefunc=max):
        """
        Do the dynamic programming iteration: iterate from tmax-1 back to 0,
        filling in the quality and decision matrices.
        
        The choicefunc argument must specify a function that can choose the
        decision that will be made from a list of possibilities, in the form
        [(quality, decision), ...].  The built-in functions max or min can be
        used, as can random.choice from the random module.
        """
        assert callable(choicefunc), "choicefunc must be a function"
        assert self.__filledterminal, "must call fill_terminal_quality before fill_quality"

        self.__create_qualarray()
        self.__create_decarray()

        for t in xrange(self.tmax-2,-1,-1):
            if t%50 == 0:
                sys.stderr.write("dynamic fill, t=%i\n" % (t) )
            self.fill_time(t, choicefunc)

        self.__filleddec = True
        self.__filledqual = True
    
    #################################################################
    # Monte Carlo Functions

    def __prob_round_value(self, val, minval, maxval):
        """
        Probabalistically round a value to the integer above or below
        For val=x.y, return x+1 with prob y and x with prob 1-y.

        [See Clark/Mangel, p. 115.]
        """
        # first, are we in-bounds?
        if val <= minval:
            return minval
        elif val >= maxval:
            return maxval
        
        # round with the appropriate probability
        whole = int(val)
        frac = val-whole
        if random.random()<frac:
            return whole+1
        else:
            return whole
    
    def prob_round(self, state):
        """
        Probabalistically round the values in a state to the integer above
        or below.
        """
        assert hasattr(state, '__iter__') # isiterable
        assert len(state) == self.__len        

        return tuple([ self.__prob_round_value(x, 0, var.size()-1)
                for x,var in zip(state, self.vars.all()) ])

    def monte_next_state(self, t, *state):
        """
        Return the decision, next state, and outcome description 
        in a Monte Carlo simulation.
        """
        assert hasattr(state, '__iter__') # isiterable
        assert len(state) == self.__len
        
        dec = self.decision0(t, state)
        outcomes = self.possibilities(self, t, *state)
        #print "--",dec
        #print t, state
        #print outcomes
        #print
        state, descr = outcomes.choose_outcome(dec)
        return dec, state, descr

    def __identity(self, x):
        return x

    def monte_carlo_run(self, start, tmax=None, dyn_time=None,
            report_step=None, report_final=None):
        """
        Do a Monte Carlo simulation of the model, starting at the given state.
        
        The dyn_time function can be used to convert times in the Monte Carlo
        simulation to corresponding times in the dynamic system.  This can be
        used to create sequential coupling in the forward simulation.  If no
        value is given, the identity function is assumed, so the time in the
        dynamic and Monte Carlo simulations are equivalent.
        """
        assert dyn_time==None or callable(dyn_time), "dyn_time must be a function or None"
        assert report_step==None or callable(report_step), "report_step must be a function or None"
        assert report_final==None or callable(report_final), "report_final must be a function or None"        
        assert self.__filleddec, "must fill in the decision array before doing Monte Carlo runs"
        
        # sort out the arguments
        t = start[0]
        state = start[1:]
        if tmax==None:
            tmax = self.tmax
        if tmax!=self.tmax and dyn_time==None:
            raise ValueError, "If you set tmax, you must give a time conversion function (dyn_time)"
        if dyn_time==None:
            dyn_time = self.__identity
        
        
        while t<tmax-1:
            dyn_t = dyn_time(t)
            dec, next_state, descr = self.monte_next_state(dyn_t, *state)
            if next_state==None:
                # dead
                break
            
            # update the Monte time.  This assumes that time passes at the
            # same rate, regardless of dyn_time().  Any complaints?
            delta_t = next_state[0] - dyn_t
            t += delta_t

            # new state
            #print dec, next_state
            state = self.prob_round(next_state[1:])
            
            if report_step != None:
                report_step(self, dec, descr, t, *state)

        else:
            descr = "endtime"
        
        if report_final != None:
            # TODO: give impossible decision?
            report_final(self, None, descr, t, *state)

    
    #################################################################
    # Outcome Distribution Functions
    
    def __addprob(self, mapping, key, prob):
        """
        Add the given probability to the given key.
        """
        k = tuple(key)
        mapping[k] = mapping.get(k, 0.0) + prob
    
    def __distrib_roundn(self, n, prob):
        """
        Recursively round and interpolate state variables 0..n.
        """
        #print ">>>", n, prob
        if n<=0:
            # recursive base-case: don't bother interpolating time
            return prob
        
        newprob = {}
        for state in prob:
            s = list(state)
            p = prob[state]
            xc = state[n]
            var = self.vars[n-1] # -1 because we have times in here too

            # check variable bounds
            if xc <= 0:
                # below 0
                s[n] = 0
                self.__addprob(newprob, s, p)
                continue
            if xc >= var.size()-1 :
                s[n] = var.size()-1
                self.__addprob(newprob, s, p)
                continue

            if var.discrete():
                # discrete variable: no need to interpolate/round
                assert int(xc)==xc, "discrete variable given non-integer value: state=" + str(state)
                self.__addprob(newprob, state, p)
                continue
        
            x = int(xc)
            if x==xc:
                # integer: no need to interpolate
                self.__addprob(newprob, state, p)
                continue

            dx = xc-x
            
            # really rounding & interpolating: add the two possibilities
            s[n] = x
            self.__addprob(newprob, s, p*(1-dx))
            s[n] = x+1
            self.__addprob(newprob, s, p*dx)
        
        return self.__distrib_roundn(n-1, newprob)
            

    def distrib_round(self, states):
        """
        Takes a continuous state.  Return the corresponding probability
        distribution for neighbouring *integer* states (as state, prob pairs).
        
        That is, round the values probabilistically, adjusting the arrival 
        probabilities accordingly.  Return a list of all outcomes, with their
        probabilities.
        """
        #assert hasattr(state, '__iter__') # isiterable
        #assert len(state) == self.__len        

        return self.__distrib_roundn(self.__len, states)

    def distrib_time(self, t):
        for state in self.vars.allvalues():
            # core stuff we need to know about this state
            n = self.__distrib[ self.index(t, state) ]
            if n==0:
                continue
            
            # get the list of possible outcomes for the decision
            dec = self.__dec[ self.index(t, state) ]
            outcomes = self.possibilities(self, t, *state)
            dec_outcomes = outcomes.decision_outcomes(dec)
            
            # add in the number that get to each next state
            dec_outcomes_round = self.distrib_round(dec_outcomes)
            for nextstate in dec_outcomes_round:
                prob = dec_outcomes_round[nextstate]
                t0 = nextstate[0]
                state0 = nextstate[1:]
                self.__distrib[ self.index(t0, state0) ] += n*prob


    def fill_distrib(self, initial_state=None, num=100, initial_function=None):
        """
        Fill in the array of probability distributions, with all individuals
        starting time in the given state.
        """
        assert self.__filleddec, "must fill in the decision array before analyzing distributions"
        assert self.__filledqual, "must fill in the quality array before analyzing distributions"
        
        self.__create_distribarray()
        # set all states to zero individuals
        for state in self.vars.allvalues():
            self.__distrib[ self.index(0, state) ] = 0

        # seed at the given state
	if initial_state!=None and initial_function!=None:
	    raise "Can't specify both initial_state and initial_function."
	elif initial_state!=None:
            assert hasattr(initial_state, '__iter__') # isiterable
            assert len(initial_state) == self.__len
            assert type(num) == int
            # set all states to zero individuals, and put individuals
	    # at one state
            for state in self.vars.allvalues():
                self.__distrib[ self.index(0, state) ] = 0
            self.__distrib[ self.index(0, initial_state) ] = num
	elif initial_function!=None:
	    assert callable(initial_function)
	    # Fill t=0 from given function
	    for state in self.vars.allvalues():
                self.__distrib[ self.index(0, state) ] = initial_function(*state)
	else:
	    raise "Must specify either initial_state or initial_function."
        
        # go through time, filling in the number that get to each state.
        for t in xrange(self.tmax-1):
	    self.distrib_time(t)


        #print self.__distrib


    #################################################################
    # File I/O functions

    def write_dec_file(self, filename):
        """
        Write the decision data out to a file called filename.
        """
        sys.stderr.write("writing decision file\n" )
        fh = file(filename, "wb")
        fh.write("Python dynmodel dec %s\n%s %s\n\0"
                % (version, self.dec_type, self.__decarraydim()) )
        self.__dec.tofile(fh)

        fh.close()

    def write_qual_file(self, filename):
        """
        Write the quality data out to a file called filename.
        """
        fh = file(filename, "wb")
        #fh.write("Python dynmodel qual %s\n%s %s\n\0"
        #        % (version, qual_type, self.__qualarraydim()) )
        sys.stderr.write("writing quality file\n" )
        self.__qual.tofile(fh)
        fh.close()

    def read_dec_file(self,filename):
        """
        Read the decision data from the given file.
        """
        sys.stderr.write("loading decision file\n" )
        fh = file(filename, "rb")

        # eat magic lines
        while True:
            ch = fh.read(1)
            if ch=="\0":
                break

        self.__dec = fromfile(fh, self.dec_type, self.__decarraydim())

        fh.close()
        self.__filleddec = True

    def read_qual_file(self,filename):
        """
        Read the quality data from the given file.
        """
        sys.stderr.write("loading quality file\n" )
        fh = file(filename, "rb")

        # eat magic lines
        #while True:
        #    ch = fh.read(1)
        #    if ch=="\0":
        #        break

        self.__qual = fromfile(fh, qual_type, self.__qualarraydim())

        fh.close()
        self.__filledqual = True
        self.__filledterminal = True

    def read_files(self, qualfile, decfile):
        """
        Read the quality and decision data from the given files.
        """
        self.read_qual_file(decfile)
        self.read_dec_file(decfile)


    #################################################################
    # Phase space functions

    def decision_subarray(self):
        import subarray
        tvar = Variable(self.tmax-1, "time", False)
        vars = [tvar] + self.vars.all()
        return subarray.SubArray(self.__dec, vars)
    
    def quality_subarray(self):
        import subarray
        tvar = Variable(self.tmax, "time", False)
        vars = [tvar] + self.vars.all()
        return subarray.SubArray(self.__qual, vars)
    
    def distrib_subarray(self):
        import subarray
        tvar = Variable(self.tmax, "time", False)
        vars = [tvar] + self.vars.all()
        return subarray.SubArray(self.__distrib, vars)
    
    def draw_regions(self, clr, slices):
        raise # dead function
        import regions
        regions.draw_regions(self, clr, slices)


