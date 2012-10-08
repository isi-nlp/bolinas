
class Edge():
    """
    stores an edge label including alignment information to the NL part 
    """
    def __init__(self,label,count=0,code=0):
        self.label = label
        self.count = int(count)
        (self.nonterminal,self.terminal) = (code==0,code==1) 
        self.align = set()
        #self.tags = dict()

    def isNonterminal(self):    return self.nonterminal
    def isTerminal(self):       return self.terminal

    def __str__(self):      return str(self.label)#return " ".join((self.label,self.lexical,str(self.count)))
    def __repr__(self):     return " ".join((self.label,str(self.count)))
    def __hash__(self):     return hash(str(self.label + str("___") + str(self.count)))
    def __eq__(self,other): return self.__hash__() == other.__hash__()

    def __getitem__(self,i):  return (self.label,self.count)[i]

    def __setitem__(self,i,v):
        if i == 0:  self.label = v
        if i == 1:  self.count = v


class RuleInstance(object):
    """
    stores an individual canonical rule instance:
    from triple -> [to triples] with a label for the rule type
    """

    def __init__(self,fromAT,toATs,ruleType):
        self.fromAT = fromAT            # tuple
        self.toATs = toATs              # [tuples]
        self.ruleType = ruleType        # string

    def __str__(self):  return repr(self.fromAT) + " -> " + repr(self.toATs)
    def __repr__(self): return repr(self.fromAT) + " -> " + repr(self.toATs)
    def __eq__(self,other): return self.__hash__() == other.__hash__()

    def __hash__(self):
        hash = 1
        for toAT in self.toATs:
            hash += toAT.__hash__() \
                    + 100 * toAT[1].__hash__() \
                    + 1000 * toAT[2].__hash__()
        hash *= 100
        hash += self.fromAT[0].__hash__() \
                + 100 * self.fromAT[1].__hash__() \
                + 1000 * self.fromAT[2].__hash__()
        return hash

class CanonicalDerivation(object):

    def __init__(self,amrs,rules=[],count=1):
        self.amrs = amrs    # [AMR] (= MRT)
        self.rules = rules  # [RuleInstance]
        self.count = count  # keeps count of derivation index

    @classmethod
    def derive(cls,parent,amr,rule):
        amrs = [amr] + parent.amrs
        rules = [rule] + parent.rules
        count = parent.count + 1
        return cls(amrs,rules,count)

    def __eq__(self,other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        tup = tuple(sorted(self.rules))
        return sum([100 * i * tup[i].__hash__() for i in range(len(tup))]) # ??

    def get_triples(self,which=0):
        return self.amrs[which]


def tibFormat(string,reverse=False):
    if not reverse:
        string = string.replace(".","-PERIOD-")
        string = string.replace("%","-PERCENTAGE-")
        string = string.replace(":","-COLON-")
        string = string.replace("@","-ATSYMBOL-")
        string = string.replace("#","-HASH-")
        string = string.replace("_","-UNDERSCORE-")
    else:
        string = string.replace("-PERIOD-",".")
        string = string.replace("-PERCENTAGE-","%")
        string = string.replace("-COLON-",":")
        string = string.replace("-ATSYMBOL-","@")
        string = string.replace("-HASH-","#")
        string = string.replace("-UNDERSCORE-","_")
        string = string.replace("-period-",".")
        string = string.replace("-percentage-","%")
        string = string.replace("-colon-",":")
        string = string.replace("-atsymbol-","@")
        string = string.replace("-hash-","#")
        string = string.replace("-underscore-","_")
    return string

def galFormat(string,reverse=False):
    if not reverse:
        string = string.replace("-","XXX")
    else:
        string = string.replace("XXX","-")
    return string

