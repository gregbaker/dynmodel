"""
Patch selection model from Dynamic State Variable Models in Ecology, Clark & Mangel.
"""

import dynmodel
from patch_param import *

def output(data, model, t):
    """
    Output the data from time t to the data filehandle.
    """
    data.write("Time period %3d\n" % (t))
    data.write("----------------------\n")
    data.write(" x  dec(x,t)  F(x,t)\n");
    data.write("----------------------\n")
    for x in range(1,xmax+1):
        data.write("%3d %5d   %8.3f\n" % ( x,
                model.decision(t, x), model.quality(t, x) ))
    data.write("----------------------\n\n")

def phi(x):
    """
    Terminal fitness function
    """
    return acap*x/(x + x0)

def possibilities(model, t, x):
    """
    Return an Outcomes object that represents all possible outcomes from this state.
    """
    
    possib = dynmodel.Outcomes(model)
    if x==0:
        # starved
        possib.otherwise(0, qual=0, descr="starved")
        return possib
    
    # visit patch 1 or 2
    for n in [1,2]:
        possib.add( # find food
                    decision=n,
                    qualgain=0,
                    prob=(1 - m[n-1]) * p[n-1],
                    nextstate=(t+1, x - cost + y[n-1]),
                    descr="findfood" )

        possib.add( # no food
                    decision=n,
                    qualgain=0,
                    prob=(1 - m[n-1]) * (1 - p[n-1]),
                    nextstate=(t+1, x - cost),
                    descr="nofood" )

    # try to reproduce, decision 3
    n=3
    if x < xrep:
        possib.add( # can't reproduce
                    decision=n,
                    qualgain=0,
                    prob=1 - m[n-1],
                    nextstate=(t+1, x - cost),
                    descr="norep" )

    elif x < xrep+c3:
        possib.add( # limited reproduction
                    decision=n,
                    qualgain=x - xrep,
                    prob=1 - m[n-1],
                    nextstate=(t+1, xrep - cost),
                    descr="norep" )

    else:
        possib.add( # full reproduction
                    decision=n,
                    qualgain=c3,
                    prob=1 - m[n-1],
                    nextstate=(t+1, x - cost-c3),
                    descr="norep" )
    
    return possib
    

def buildmodel():
    varx = dynmodel.Variable(xmax+1, "energy")

    model = dynmodel.Model(tmax+1, [varx],
            terminal_quality=phi,
            possible_outcomes=possibilities,
            )
    return model

def main():
    model = buildmodel()

    model.fill_terminal_quality()
    model.fill_quality(max)

    data = open("out0.txt", 'w')
    for t in range(tmax-1, 0, -1):
        output(data, model, t)
    data.close()
    
    model.write_dec_file("patchdec.data")

if __name__=='__main__':
    main()

