#import copy
#import re, sys
from collections import defaultdict
#from Queue import Queue

from data_structures import CanonicalDerivation, Edge, RuleInstance

class CanonicalParser(object):

    def __init__(self,s):
        """
        Takes a sentence and learns a canonical derivation according to the simple grammar defined below.
        """

        derivs_cur = set()
        derivs_fail = set()
        self.derivs_done = set()
        derivs_all = set()

        self.s = s

        # Add the full AMR to start with
        derivs_cur.add(CanonicalDerivation([s['mrt']]))

        while len(derivs_cur) > 0:
            derivation = derivs_cur.pop()
            if len(derivation.get_triples()) == 1 and derivation.get_triples()[0][1].isNonterminal():
                self.derivs_done.add(derivation)
                derivs_all.add(derivation)
            else:
                deriv = False

                for rule in [self.applyDelex,self.applySL,self.applySW, \
                        self.applySO,self.applyCircle,self.applyJointHit,self.applyElongate]:
                    deriv = rule(derivation)
                    if deriv: break

                # If we don't learn anything, add this derivation to the failures
                if not deriv:
                    derivs_fail.add(derivation)
                else:
                    # If we've seen this derivation before, don't go there again
                    if deriv not in derivs_all:
                        derivs_cur.add(deriv)
                derivs_all.add(deriv)
                
        self.derivs_done = list(self.derivs_done)
        self.derivs_fail = list(derivs_fail)
        #print "Failed derivations:   ", len(derivs_fail)
        print "Complete derivations: ", len(self.derivs_done)
        
        """
        # Print the failed derivations to see what went wrong
        for d in self.derivs_fail:
            print "Failed derivation: "
            print d.get_triples()
        """


    def applyDelex(self,d):
        triples = d.get_triples()
        for i in xrange(len(triples)):
            (a,b,c) = triples[i]
            if b.isTerminal():
                ntLabel,tmp = b[0].split(":",1)
                nrf = (a,Edge(ntLabel,d.count),c)
                nrt = [triples[i]]

                new_mrt = list(triples)
                new_mrt[i] = nrf # replace triple with new triple

                new_rule = RuleInstance(nrf,nrt,'DL')
                return CanonicalDerivation.derive(d,new_mrt,new_rule)
        return False

    def applySL(self,d):
        """
        Search for any node with one occurence as p1 and one as p2 only.
        Combine these two by removing that node and merging the edges.
        """
        triples = d.get_triples()
        ANodes = defaultdict(int)
        BNodes = defaultdict(int)
        for (a,b,c) in triples:
            ANodes[a] += 1
            BNodes[c] += 1
            
        for a in ANodes.keys():
            if ANodes[a] == 1 and BNodes[a] == 1:
                # we have an edge that we can shorten: remove (x,X,a) and (a,X,z) for (x,Y,z)
                nrf = [None,Edge('*',d.count),None] # new rule from
                nrt = [0,0]                         # new rule to
                new_amr = list(triples)
                for i in xrange(len(triples)):
                    at = triples[i]
                    if at[0] == a and at[2] != a:
                        nrf[2] = at[2]
                        nrt[1] = at
                    elif at[2] == a and at[0] != a:
                        nrf[0] = at[0]
                        nrf[1][0] = at[1][0]
                        nrt[0] = at
                        index = i

                if nrt[0][1].isNonterminal() and nrt[1][1].isNonterminal():
                    new_amr[index] = tuple(nrf)
                    new_amr.remove(nrt[1])
                    new_rule = RuleInstance(tuple(nrf),nrt,'SL')
                    return CanonicalDerivation.derive(d,new_amr,new_rule)
        return False
    
    def applySW(self,d):
        """
        Search for any multiple edges (a-X-b) and merge two of these
        """
        triples = d.get_triples()
        Nodes = defaultdict(int)
        for (a,b,c) in triples:
            Nodes[(a,c)] += 1
            
        for (a,c) in Nodes.keys():
            if Nodes[(a,c)] > 1:
                # We have one edge that we can remove: remove (a,X,b) and (a,Y,b) for (a,Y,b)
                # If more than two, we can remove any one of these, given any other one of these
                for i in xrange(len(triples)):
                    candidate = triples[i]
                    (x,y,z) = candidate
                    if x == a and z == c and y.isNonterminal():
                        for j in xrange(i+1,len(triples)):
                            candidate2 = triples[j]
                            (k,l,m) = candidate2
                            if k == x and m == z and l.isNonterminal() and candidate != candidate2:
                                nrf = (k,Edge(y[0],d.count),m)
                                nrt = [candidate,candidate2]
                                new_amr = list(triples)
                                new_amr[i] = nrf
                                del new_amr[j]
                                new_rule = RuleInstance(nrf,nrt,'SW')
                                return CanonicalDerivation.derive(d,new_amr,new_rule)
        return False

    def applySO(self,d):
        """
        Search for any split a-X-b,a-Y-c where c is a leaf node
        Remove a-Y-c and let it be generated by a-X-b
        """
        triples = d.get_triples()
        Leaves = defaultdict(int)
        Branches = defaultdict(int)
        for (a,b,c) in triples:
            Leaves[c] += 1
            Branches[a] += 1

        # If leaves[b] == 1 and branches[a] > 1 we can remove the (a,X,b) edge using SO
        for i in xrange(len(triples)):
            candidate = triples[i]
            (a,b,c) = candidate
            if Leaves[c] == 1 and Branches[a] > 1 and Branches[c] == 0 and b.isNonterminal():
                for j in xrange(len(triples)):
                    candidate2 = triples[j]
                    (x,y,z) = candidate2
                    if x == a and z != c and y.isNonterminal():
                        # Depending on the grammar it would make sense to install a clause here
                        # which determines the 'surviving' edge based on some implicit ordering
                        nrf = (x,Edge(y[0],d.count),z)
                        nrt = [candidate2,candidate]
                        rulename = 'OL' # short for open-left
                        new_amr = list(triples)
                        new_amr[j] = nrf
                        del new_amr[i]
                        new_rule = RuleInstance(nrf,nrt,rulename)
                        return CanonicalDerivation.derive(d,new_amr,new_rule)
        return False

    def applyJointHit(self,d):
        """
        edge A-B becomes edges A-C and B-C in reverse
        """

        child = defaultdict(set)
        parent = defaultdict(set)

        triples = d.get_triples()
        for trip in triples:
            (a,b,c) = trip
            child[a].add(trip)
            parent[c].add(trip)

        for i in xrange(len(triples)):
            candidate1 = triples[i]
            (a,x,c) = candidate1
            if len(child[c]) == 0 and len(parent[c]) == 2 and x.isNonterminal():
                for candidate2 in parent[c]:
                    (b,y,tmp) = candidate2
                    if y.isNonterminal() and b != a: # we know that c == tmp
                        wrongWay = False
                        for check in child[b]:
                            # optional (attempts to avoid generating looped structures)
                            (k,l,m) = check
                            if m == a:  wrongWay = True
                        if not wrongWay:
                            # We found a candidate to remove (a,x,c) (b,y,c) down to (a,?,b)
                            # Now, let's iterate so that we can find the suitable edges (with labels)
                            nrf = (a,Edge('*',d.count),b)
                            nrt = [candidate1,candidate2]
                            new_amr = list(triples)
                            new_amr[i] = nrf
                            new_amr.remove(candidate2)
                            new_rule = RuleInstance(nrf,nrt,'JH')
                            return CanonicalDerivation.derive(d,new_amr,new_rule)
        return False

    def applyElongate(self,d):
        """
        A->B becomes A->B->C in reverse
        """

        child = defaultdict(set)
        parent = defaultdict(set)

        triples = d.get_triples()
        for trip in triples:
            (a,b,c) = trip
            child[a].add(trip)
            parent[c].add(trip)
        
        for i in xrange(len(triples)):
            candidate1 = triples[i]
            (b,x,c) = candidate1
            if len(child[c]) == 0 and len(parent[c]) == 1 and x.isNonterminal():
                for candidate2 in parent[b]:
                    (a,y,tmp) = candidate2 
                    if y.isNonterminal(): # we already know tmp == b
                        # We found a candidate to remove (a,y,b,x,c) down to (a,y,b)
                        nrf = (a,Edge(y[0],d.count),b)
                        nrt = [candidate2,candidate1]
                        new_amr = list(triples)
                        new_amr[i] = nrf
                        new_amr.remove(candidate2)
                        new_rule = RuleInstance(nrf,nrt,'LL')
                        return CanonicalDerivation.derive(d,new_amr,new_rule)
        return False
    
    def applyCircle(self,d):
        """
        A->B becomes A->B->B (circle) in reverse
        """

        parent = defaultdict(set)

        triples = d.get_triples()
        for i in xrange(len(triples)):
            (a,b,c) = triples[i]
            parent[c].add((i,triples[i]))
        
        for i in xrange(len(triples)):
            candidate1 = triples[i]
            (a,b,c) = candidate1
            if a == c and b.isNonterminal():
                for index,candidate2 in parent[c]:
                    (x,y,z) = candidate2
                    if y.isNonterminal():
                        # We found a candidate to remove (x,y,a,b,a) down to (x,y,a)
                        nrf = (x,Edge(y[0],d.count),z)
                        nrt = [candidate2,candidate1]
                        new_amr = list(triples)
                        new_amr[index] = nrf
                        del new_amr[i]
                        new_rule = RuleInstance(nrf,nrt,'CC')
                        return CanonicalDerivation.derive(d,new_amr,new_rule)
        return False
