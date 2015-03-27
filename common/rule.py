from common.hgraph.hgraph import Hgraph
import math

class Rule(object):

    # TODO: There is probably other code shared between the two rule classes 
    #   that can be in this base class.

    def __str__(self):
        weight = self.weight if self.logprob else math.exp(self.weight)
        
        if isinstance(self.rhs1, Hgraph):
            rhs1string = self.rhs1.to_string()
        else: 
            rhs1string = " ".join([str(x) for x in self.rhs1])
        if self.rhs2:
            if isinstance(self.rhs2, Hgraph):
                rhs2string = self.rhs2.to_string()
            else: 
                rhs2string = " ".join([str(x) for x in self.rhs2])
            return "%s -> %s | %s ; %.10f" % (self.symbol, rhs1string, rhs2string, weight)
        else: 
            return "%s -> %s ; %.10f" % (self.symbol, rhs1string, weight)
       
