import copy
from collections import defaultdict


class AMRtriple(object):

    def __getitem__(self,k):    return (self.a,self.b,self.c)[k]

    def __init__(self,a,b,c,code=0):
        '''
        From, Edge, To
        Code: 0 = nonterminal, 1 = delex, 2 = terminal
        '''
        
        self.a = str(a)     # Irrelevant Vertex
        self.b = b          # Edge
        self.c = str(c)     # Irrelevant Vertex

        self.align = set()
        self.span = []
        self.tags = dict()

        self.code = code

        (self.nonterminal,self.delexicalised,self.terminal) = (code==0,code==1,code==2) 

    def isNonterminal(self):    return self.nonterminal

    def isTerminal(self):       return self.terminal

    def isDelexicalised(self):  return self.delexicalised

    def isNotTerminal(self):    return self.nonterminal or self.delexicalised

    def get(self):
        return (self.a,self.b,self.c)

    def __hash__(self):
        return hash(self.a) + 100 * hash(self.b) + 10000 * hash(self.c)

    def __eq__(self,other): return self.__hash__() == other.__hash__()

    def __str__(self): 
        return ' '.join((str(self.a),str(repr(self.b)),str(self.c)))

    def __repr__(self):
        return str(self)

class Edge():

    def __init__(self,label,lexical,count=0):
        self.label = label
        self.lexical = lexical
        self.count = int(count)

    def __str__(self):      #return " ".join((self.label,self.lexical,str(self.count)))
        if self.count == 0: return str(self.label) + ':' + str(self.lexical)
        else:               return str(self.label) + '(' + str(self.count) + ')'
    def __repr__(self):     return " ".join((self.label,self.lexical,str(self.count)))

    #def __repr__(self): return str(self.label) + str(self.count)

    def __hash__(self):
        #return 1000 * self.count + hash(str(self.label)) + hash(str(self.lexical))
        return hash(str(self.label)) + hash(str(self.lexical))
        #return hash(str(self.label)) + hash(self.count) + hash(str(self.lexical) + self.count)
    
    def __eq__(self,other):
        return self.__hash__() + self.count.__hash__() == other.__hash__() + other.count.__hash__()

    def __getitem__(self,i):  return (self.label,self.lexical,self.count)[i]

    def __setitem__(self,i,what):
        if i == 0:
            self.label = what
        if i == 1:
            self.lexical = what
        if i == 2:
            self.count = what

class AMR(object):

    def __init__(self,triples=set(),root=[]):
        self.triples = list(triples)
        self.root = root 

    def add_triple(self,triple):
        self.triples.append(triple)

    def add_root(self,root):
        self.root.append(root)

    def remove_triple(self,triple):
        x = len(self.triples)
        for i in range(len(self.triples)):
            if self.triples[i] == triple:
                del self.triples[i]
                break
        y = len(self.triples) + 1
        assert x == y, "Deletion remove none or more than one elements"

    def replace_triple(self,old,new):
        replace = False
        for i in range(len(self.triples)):
            if self.triples[i] == old:
                self.triples[i] = new
                replace = True
                break
        assert replace == True, "Replace Triple failed to find triple"

    def clone(self):
        '''
        Return a (semi-)deep copy of the AMR.
        We keep triple instances as they were - no need for additional copies
        '''
        new = AMR()
        new.root = copy.copy(self.root)
        for tr in self.triples:
            new.add_triple(tr)
        return new

class TripleRule(object):

    def __init__(self,fromAT,toATs,ruleType):
        self.fromAT = fromAT            # AMRtriple
        self.toATs = toATs              # [AMRtriple]
        self.ruleType = ruleType

    def __str__(self):  return repr(self.fromAT) + " -> " + repr(self.toATs)

    def __repr__(self): return repr(self.fromAT) + " -> " + repr(self.toATs)

    def __eq__(self,other): return self.__hash__() == other.__hash__()

    def __hash__(self):
        hash = 1
        for toAT in self.toATs:
            hash += toAT.__hash__()
        hash *= 100
        hash += self.fromAT.__hash__()
        return hash

class CanonicalDeriv(object):

    def __init__(self,amrs,rules=[],count=1):
        self.amrs = amrs    # [AMR]
        self.rules = rules  # [TripleRule]
        self.count = count

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
        return sum([100 * i * tup[i].__hash__() for i in range(len(tup))])

    def get_triples(self,which=0):
        return self.amrs[which].triples

    def finalize(self):
        '''
        Anonymise tree edges for full symmetry. Leave leaf edges untouched except for fixing the ordering in instance edges
        Only update the rules, not the stored AMRs (not sure if we'll need these anyway)
        '''
        for i in range(len(self.rules)):
            self._clean([self.rules[i].fromAT])
            self._clean(self.rules[i].toATs)

    def _clean(self,ATs):
        pass
        #for at in ATs:
        #    if at.isNonterminal():
        #        if not at.b.count is None:
        #            at.b.label = at.b.label + str(at.b.count)
        #            at.b.count = None

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

