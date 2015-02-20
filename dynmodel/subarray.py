import csv, copy, itertools
import regions

def ceil_div(a,b):
    """
    Return ceiling of a/b, with a,b integers.
    """
    if a%b == 0:
        return a//b
    else:
        return a//b + 1


class SubArray(object):
    """
    A subarray object.  This will likely be created from either the
    decision or quality array of a model.
    
    This is mostly just a wrapper of a numarray array.  It avoid copying
    the data in memory if at all possible.
    """
    
    def __init__(self, array, vars):
        assert len(vars)==len(array.shape), "number of dimensions in array and number of variables are different"
        self.__data = array.view()
        self.__vars = vars
        self.__dim = len(self.__vars)

        self.__slices = []
        for v in vars:
            self.__slices.append( (0, v.size(), 1) )
            
    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__, self.__data, self.__vars)

    def __str__(self):
        #sl = [slice(s,e,st) for s,e,st in self.__slices]
        return str( self.__data )

    def __getitem__(self, slices):
        #print slices
        if type(slices)==int or type(slices)==slice:
            # wish away the dim=1 case
            slices = [slices]

        assert len(slices)==self.__dim, "number of slices doesn't match number of array dimensions"
        
        # copy and apply the new slices
        new = copy.copy(self)
        new.__vars = self.__vars[:]
        new.__slices = self.__slices[:]
        new.__data = self.__data[slices]

        # update our internal list of slices
        for i in xrange(new.__dim-1, -1, -1):
            sl = slices[i]
            
            if type(sl)==int:
                # simple integer indexing
                del new.__vars[i]
                del new.__slices[i]
                new.__dim -= 1
            else:
                assert type(sl)==slice, "Must give slices or integer indices for each dimension."
                # Apply a slice to this dimension.
                
                # *0 is the current slice for this dimension
                start0, end0, stride0 = self.__slices[i]
                # *1 is the (normalized) slice that's being applied
                length = ceil_div(end0-start0, stride0)
                start1, end1, stride1 = sl.indices(length)
                assert stride1>0, "can't handle negative strides in slices"
                
                # now, we compose the two to get the new slice on this
                # dimension.
                start = start0 + stride0*start1
                end = min(end0, start0 + stride0*end1)
                stride = stride0 * stride1
                
                new.__slices[i] = (start, end, stride)
        
        if new.__dim == 0:
            # special case for true indexing
            return new.__data
        else:
            return new

    def xvals(self, dim):
        """
        Return an iterable that yields the values along the given
        dimension.
        """
        return xrange( *self.__slices[dim] )

    def vals(self, dim):
        """
        Return a list of the values along the given dimension.
        """
        return list(self.xvals(dim))

    def label(self, dim):
        """
        Return the description for the given dimension.
        """
        return self.__vars[dim].descr()


    def __allind(self, num):
        """
        Recursive algorithm to generate all possible index values for this
        array.
        """
        if num==self.__dim-1:
            for v in xrange( *self.__slices[num] ):
                yield (v,)
            return
        
        for v in xrange( *self.__slices[num] ):
            for r in self.__allind(num+1):
                yield (v,) + r

    def allindices(self):
        """
        Return an iterator that gives all possible index values.
        """
        for v in self.__allind(0):
            yield v



    def t(self):
        return self.__slices
    def shape(self):
        """
        Return a tuple with the size of each dimension.
        """
        return self.__data.shape

    def range(self, dim):
        """
        Return the min and max values for the given dimension
        """
        start, end, stride = self.__slices[dim]
        # TODO: max val is actually <= end
        return (start, end)


    def output(self, filename):
        assert self.__dim in [1,2], "can only output 1- or 2-dimensional slices"
        
        writer = csv.writer(file(filename, "wb"))
        
        if self.__dim == 2:
            legend = [ self.__vars[0].descr() + " \\ " + self.__vars[1].descr() ]
            legend += self.vals(1)
            writer.writerow(legend)
        
            for ind, key in itertools.izip( xrange(self.__data.shape[0]),
                                            self.xvals(0) ):
                writer.writerow( [key] + list(self.__data[ind]) )
        
        else: # self.__dim == 1:
            writer.writerow([self.__vars[0].descr(), "Value"])
            for ind, key in itertools.izip( xrange(self.__data.shape[0]),
                                            self.xvals(0) ):
                writer.writerow( [key, self.__data[ind]] )

    def draw_space(self, filename, colour,
            sqwidth=10, sqheight=10
            ):
        assert self.__dim==2, "can only create space diagram 2-dimensional slices"

        regions.draw_regions(self, filename, colour, sqwidth=sqwidth, sqheight=sqheight)
