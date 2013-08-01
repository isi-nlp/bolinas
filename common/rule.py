from common.hgraph.hgraph import Hgraph

class Rule(object):

    # TODO: There is probably other code shared between the two rule classes 
    #   that can be in this base class.

    def __str__(self):
        if isinstance(self.rhs1, Hgraph):
            rhs1string = self.rhs1.to_string()
        else: 
            rhs1string = " ".join(self.rhs1)
        if self.rhs2:
            if isinstance(self.rhs2, Hgraph):
                rhs2string = self.rhs2.to_string()
            else: 
                rhs2string = " ".join(self.rhs2)
            return "%s -> %s | %s ; %f" % (self.symbol, rhs1string, rhs2string, self.weight)
        else: 
            return "%s -> %s ; %f" % (self.symbol, rhs1string, self.weight)
       
