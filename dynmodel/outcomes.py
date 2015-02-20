import random
import model as modelmodule
if __debug__:
    import operator
    import types

class Outcomes(object):
    """
    Set of all possible outcomes that can occur from a given state.
    
    The "nochoice" parameters are used for a state with no valid decisions
    that can be made.  The decision is a dummy value of the correct type;
    the quality should be *worse* than dead, since this decision should *never*
    be made.
    """
    def __init__(self, model, nochoice_qual=-1.0, nochoice_dec=-1, nochoice_descr="impossible"):
        assert isinstance(model, modelmodule.Model), "First argument must be a Model instance."
        assert operator.isNumberType(nochoice_qual), "nochoice_qual must be a number"
        # TODO: check that nochoice_dec has right type?
        assert type(nochoice_descr) in types.StringTypes, "nochoice_descr must be a string"
        
        self.__model = model
        self.__outcomes = []
        self.__choices = set()
        self.__otherqual = {}
        self.__otherdescr = {}
        # what do we do if there are no choices that can be made?
        # this *should* never be reached in a simulation.
        self.__nochoicequal = nochoice_qual
        self.__nochoicedec = nochoice_dec
        self.__nochoicedescr = nochoice_descr
    
    def add(self, **outcome):
        """
        Add a new possible outcome to this state.
        
        These fields are required:
        decision=d
            The decision that's made to lead to this outcome.
        nextstate=(t, ...)
            The new state if this is the outcome.  It should be a tuple 
            with the new time, and values for the other state variables in
            the model.
        prob=0.1
            The probability of this outcome happening, given that the
            corresponding decision is made.
        descr="search"
            A description of this outcome which can be used in a simulation
            to describe it.
        qualgain=0.0
            The gain in "quality" if this outcome occurs.  Use 0 for none.
        
        Other fields can be given and used if the outcome_expected() function
        is overridden to make use of them.
        """
        # TODO: check variables in range for nextstate?
        assert outcome.has_key('decision'), "Must give a 'decision' in outcome."
        assert outcome.has_key('nextstate'), "Must give a 'nextstate' in outcome."
        assert type(outcome['nextstate'])==types.TupleType, "The 'nextstate' must be a tuple containing the next time and state values: (t, x, y, ...)"
        assert outcome.has_key('prob'), "Must give a occurence probability (prob) in outcome"
        assert operator.isNumberType(outcome['prob']), "The probability (prob) must be a number from 0 to 1"
        assert 0<=outcome['prob']<=1, "The probability (prob) must be a number from 0 to 1"
        assert outcome.has_key('descr'), "Must give a short description (descr) in outcome."
        assert type(outcome['descr']) in types.StringTypes, "Description (descr) must be a string"
        assert outcome.has_key('qualgain'), "Must give a quality/fitness increate (qualgain) in outcome: 0 for none."
        assert operator.isNumberType(outcome['qualgain']), "The quality gain (qualgain) must be a number"

        # record the possible outcome
        self.__outcomes.append( outcome )

        # maintain the "otherwise" case if necessary
        dec = outcome['decision']
        self.__choices.add(dec)
        if not self.__otherqual.has_key(dec):
            # set defaults for the "otherwise" case
            self.__otherqual[dec] = 0.0
            self.__otherdescr[dec] = "die"

            
    def otherwise(self, dec, qual=0.0, descr="die"):
        """
        Set the details for the "otherwise" case in this decision.
        
        The "otherwise" case is what happens if none of the given outcomes
        occur (with probability 1-sum(outcome[probs])).  In particular,
        we need to know what quality/fitness does it has, and what to call it?
        """
        assert operator.isNumberType(qual), "quality must be a number"
        assert type(descr) in types.StringTypes, "description must be a string"
        
        self.__choices.add(dec)
        self.__otherqual[dec] = qual
        self.__otherdescr[dec] = descr
    
        
    def outcome_expected(self, outcome):
        """
        The expected quality contribution from a single outcome.
        """
        return outcome['qualgain'] + outcome['prob']*( \
                self.__model.interp(*outcome['nextstate']) )


    def choices(self):
        """
        Return the set of possible decision choices from this state.
        """
        return self.__choices


    def all_expected(self):
        """
        Return a dictionary of the possible decisions and their corresponding
        expected quality values.
        """
        if len(self.__choices) == 0:
            # we have no options: return the dummy values
            return {self.__nochoicedec: self.__nochoicequal}
        
        # build dictionary
        expect = {}
        totalprob = {}
        for dec in self.__choices:
            expect[dec] = 0.0
            totalprob[dec] = 0.0
        
        # count up expectation from given outcomes
        for outcome in self.__outcomes:
            dec = outcome['decision']
            expect[dec] += self.outcome_expected(outcome)
            totalprob[dec] += outcome['prob']
        
        # add in the "otherwise" cases
        for dec in self.__choices:
            assert totalprob[dec] <= 1.0000001, "Total probabilities for decision "+str(dec)+" are >1.0."
            expect[dec] += (1-totalprob[dec]) * self.__otherqual[dec]

        return expect

    def __addprob(self, mapping, key, prob):
        """
        Add the given probability to the given key.
        """
        mapping[key] = mapping.get(key, 0.0) + prob
    

    def decision_outcomes(self, dec):
        """
        Return a dictionary of the possible outcomes, assuming the decision dec has
        been made.
        
        Returns outcomes {(nextstate: prob), ...} for the decision.  The
        'otherwise' case is ignored.
        """
        
        possib = {}
        for outcome in self.__outcomes:
            if outcome['decision']==dec:
                # an outcome that goes with this decision...
                self.__addprob(possib, outcome['nextstate'], outcome['prob'] )
        
        return possib


    def choose_outcome(self, dec):
        """
        Randomly select one of the outcomes, assuming the decision dec has
        been made.
        
        Returns (nextstate, descr) for the selected outcome.  If the 'otherwise'
        case is selected, nextstate is None.
        """
        rand = random.random()
        prob = 0.0

        for outcome in self.__outcomes:
            if outcome['decision']==dec:
                # an outcome that goes with this decision...
                prob += outcome['prob']
                if rand < prob:
                    return outcome['nextstate'], outcome['descr']

        # if we get this far, it's the 'otherwise' case.
        try:
            return None, self.__otherdescr[dec]

        except KeyError:
            # an impossible choice has been made
            # [left as an exception since this should be *very* rare: speed
            # up other cases instead]
            return None, self.__nochoicedescr

    def __str__(self):
        return "[" + ",\n".join(
                    [str(out)+" "+str(self.outcome_expected(out)) for out in self.__outcomes]
                ) + "]"





