import copy
import re, sys
from collections import defaultdict
from Queue import Queue

from lib.amr.amr import Amr, Dag
from data_structures import AMR, AMRtriple, CanonicalDeriv, TripleRule, Edge

class CanonicalGrammar(object):

    def __init__(self,s):
        '''
        Takes a sentence (containing all relevant info as below) and learns a set of canonical derivations according to the simple grammar defined below.
        
        sentence = ['empty',
                    'plain_text',
                    'plain_constituentstring',
                    'plain_amr',
                    'plain_constituentparse',
                    'plain_typed',
                    'plain_edg2dep',
                    'plain_edg2str',
                    'tokens',
                    'amr',
                    'depTree',
                    'depParent']

                    Rules:
                    a-X-b -> a-Y-Z-b        = split_long
                    a-X-b -> a-Y-b a-Z-b    = split_wide
                    a-X-b -> a-X-b a-Z-c    = split_open
                    a-X-b -> a-(actual)-b   = change_tag
                    a-X-b -> something complicated = hyperedge
                    '''

        '''
        Define some containers for derivations

        '''
        derivs_cur = set()
        derivs_fail = set()
        self.derivs_done = set()
        derivs_all = set()

        self.s = s

        """
        Edge ordering for AMR style graphs.
        """
        order = ['ROOT','ARG0','ARG1','ARG2','ARG3','ARG0-of','ARG1-of','ARG2-of','ARG3-of','op1','op2','op3','mod','domain']
        self.degree = defaultdict(int)
        for i in range(len(order)):
            self.degree[order[i]] = len(order) - i

        # Add the full AMR to start with
        derivs_cur.add(CanonicalDeriv([s['delexAMR']]))

        while len(derivs_cur) > 0:
            #print len(derivs_cur), len(derivs_all)
            derivation = derivs_cur.pop()
            if len(derivation.get_triples()) == 1 and derivation.get_triples()[0].isNotTerminal():
                #if len(derivation.rules) > 0: # We currently only play with AMRs that contain more than one single rule
                self.derivs_done.add(derivation)
                derivs_all.add(derivation)
            else:
                # Apply derivation rules

                derivs = self.applyDelex(derivation,[])
                if len(derivs) == 0:
                    derivs = self.applySL(derivation,derivs)
                if len(derivs) == 0:
                    derivs = self.applySW(derivation,derivs)
                if len(derivs) == 0:
                    derivs = self.applySO(derivation,derivs)
                if len(derivs) == 0:
                    derivs = self.applyJointHit(derivation,derivs)
                if len(derivs) == 0:
                    derivs = self.applyElongate(derivation,derivs)
                if len(derivs) == 0:
                    derivs = self.applyCircle(derivation,derivs)

                # If we don't learn anything, add this derivation to the failures
                if len(derivs) == 0:
                    derivs_fail.add(derivation)

                
                for deriv in derivs:
                    # If we've seen this derivation before, don't go there again
                    if deriv not in derivs_all:
                        derivs_cur.add(deriv)
                    derivs_all.add(deriv)
                
        self.derivs_done = list(self.derivs_done)
        self.derivs_fail = list(derivs_fail)
        #print "Failed derivations:   ", len(derivs_fail)
        #print "Complete derivations: ", len(self.derivs_done)
        
        for deriv in self.derivs_done: deriv.finalize()

        # Print the failed derivations to see what went wrong
        for d in self.derivs_fail:
            print "Failed derivation: "
            print d.get_triples()

    def applySL(self,d,derivs):
        '''
        Search for any node with one occurence as p1 and one as p2 only.
        Combine these two by removing that node and merging the edges.
        '''
        triples = d.get_triples()
        ANodes = defaultdict(int)
        BNodes = defaultdict(int)
        for (a,b,c) in [t.get() for t in triples]:
            ANodes[a] += 1
            BNodes[c] += 1
            
        for a in ANodes.keys():
            if ANodes[a] == 1 and BNodes[a] == 1:
                '''
                We have an edge that we can shorten: remove (x,X,a) and (a,X,z) for (x,Y,z)
                '''
                nrf = AMRtriple(None,Edge('*','*',d.count),None) # new rule from
                nrt = [0,0]            # new rule to
                new_amr = d.amrs[0].clone()
                for at in triples:
                    if at.a == a:
                        if at.c == a:
                            print "LOOPY EDGE"
                            sys.exit(-1)
                            return None
                        nrf.c = at.c
                        nrt[1] = at
                    elif at.c == a:
                        nrf.a = at.a
                        nrf.b[0] = at.b[0]
                        nrt[0] = at

                if nrt[0].isNotTerminal() and nrt[1].isNotTerminal():
                    new_amr.replace_triple(nrt[0],nrf)
                    new_amr.remove_triple(nrt[1])
                    new_rule = TripleRule(nrf,nrt,'SL')
                    derivs.append(CanonicalDeriv.derive(d,new_amr,new_rule))
                    return derivs
        return derivs        
    
    def applySW(self,d,derivs):
        '''
        Search for any multiple edges (a-X-b) and merge two of these
        '''
        triples = d.get_triples()
        Nodes = defaultdict(int)
        for (a,b,c) in [t.get() for t in triples]:
            Nodes[(a,c)] += 1
            
        for (a,c) in Nodes.keys():
            if Nodes[(a,c)] > 1:
                '''
                We have one edge that we can remove: remove (a,X,b) and (a,Y,b) for (a,Y,b)
                If more than two, we can remove any one of these, given any other one of these
                '''
                for candidate in triples:
                    (x,y,z) = candidate.get()
                    if x == a and z == c and candidate.isNotTerminal():
                        #candidate_amr = d.amrs[0].clone()
                        #triples2 = candidate_amr.triples
                        for candidate2 in triples: # (x,y,z) removed already
                            (k,l,m) = candidate2.get()
                            if k == x and m == z and candidate2.isNotTerminal() and candidate != candidate2:
                                nrf = AMRtriple(k,Edge('*','*',d.count),m)
                                if self.degree[y[0]] >= self.degree[l[0]]:
                                    nrf.b[0] = y[0]
                                    nrt = [candidate,candidate2]
                                else:                                       
                                    nrf.b[0] = l[0]
                                    nrt = [candidate2,candidate]
                                new_amr = d.amrs[0].clone()
                                new_amr.remove_triple(nrt[1])
                                new_amr.replace_triple(nrt[0],nrf)
                                new_rule = TripleRule(nrf,nrt,'SW')
                                derivs.append(CanonicalDeriv.derive(d,new_amr,new_rule))
                                return derivs
        return derivs

    def applySO(self,d,derivs):
        '''
        Search for any split a-X-b,a-Y-c where c is a leaf node
        Remove a-Y-c and let it be generated by a-X-b
        '''
        triples = d.get_triples()
        Leaves = defaultdict(int)
        Branches = defaultdict(int)
        for (a,b,c) in [t.get() for t in triples]:
            Leaves[c] += 1
            Branches[a] += 1

        '''
        If leaves[b] == 1 and branches[a] > 1 we can remove the (a,X,b) edge using SO
        '''
        for candidate in triples:
            (a,b,c) = candidate.get()
            if Leaves[c] == 1 and Branches[a] > 1 and Branches[c] == 0 and candidate.isNotTerminal():
                #candidate_amr = (d.amrs[0]).clone()
                #triples2 = candidate_amr.triples
                for candidate2 in triples:     
                    (x,y,z) = candidate2.get()
                    if x == a and z != c and candidate2.isNotTerminal():
                        nrf = AMRtriple(x,Edge('*','*',d.count),z)
                        if self.degree[b[0]] >= self.degree[y[0]]:  
                            nrf.b[0] = b[0] 
                            nrt = [candidate,candidate2]
                            rulename = 'OR'
                        else:                                       
                            nrf.b[0] = y[0]
                            nrt = [candidate2,candidate]
                            rulename = 'OL'
                        new_amr = d.amrs[0].clone()
                        new_amr.remove_triple(nrt[1])
                        new_amr.replace_triple(nrt[0],nrf)
                        new_rule = TripleRule(nrf,nrt,rulename)
                        derivs.append(CanonicalDeriv.derive(d,new_amr,new_rule))
                        return derivs
        return derivs

    def applyDelex(self,d,derivs):
        triples = d.get_triples()
        for candidate in triples:
            (a,b,c) = candidate.get()
            if candidate.isTerminal():
                x = Edge(b[0],'*',d.count)
                nrf = AMRtriple(a,x,c,1)
                nrt = [candidate]
                new_amr = (d.amrs[0]).clone()
                new_amr.replace_triple(candidate,nrf)
                new_rule = TripleRule(nrf,nrt,'DL')
                derivs.append(CanonicalDeriv.derive(d,new_amr,new_rule))
                return derivs
        return derivs

    def applyJointHit(self,deriv,derivs):
        '''
        Special rule to allow hyperedges in the canonical grammar
        (a-b) goes to (a-b) (a-c) (b-c) (c-d) (b-e)
        '''

        child = defaultdict(set)
        parent = defaultdict(set)

        triples = deriv.get_triples()
        for trip in triples:
            (a,b,c) = trip.get()
            child[a].add(trip)
            parent[c].add(trip)

        for candidate1 in triples:
            (a,x,c) = candidate1.get()
            if len(child[c]) == 0 and len(parent[c]) == 2 and candidate1.isNotTerminal():
                for candidate2 in parent[c]:
                    (b,y,d) = candidate2.get()
                    if candidate2.isNotTerminal() and b != a:
                        wrongWay = False
                        for check in child[b]:
                            (k,l,m) = check.get()
                            if m == a:  wrongWay = True
                        if not wrongWay:
                            '''
                            We found a candidate to remove (a,b,c,d,e) down to (a,b)
                            Now, let's iterate so that we can find the suitable edges (with labels)
                            '''
                            nrf = AMRtriple(a,Edge(x[0],'*',deriv.count),b)
                            nrt = [candidate1,candidate2]
                            new_amr = (deriv.amrs[0]).clone()
                            new_amr.replace_triple(nrt[0],nrf)
                            new_amr.remove_triple(nrt[1])
                            new_rule = TripleRule(nrf,nrt,'JH')
                            derivs.append(CanonicalDeriv.derive(deriv,new_amr,new_rule))
                            return derivs
        return derivs

    def applyElongate(self,deriv,derivs):
        '''
        A->B becomes A->B->C in reverse
        '''

        child = defaultdict(set)
        parent = defaultdict(set)

        triples = deriv.get_triples()
        for trip in triples:
            (a,b,c) = trip.get()
            child[a].add(trip)
            parent[c].add(trip)
        
        for candidate1 in triples:
            (b,x,c) = candidate1.get()
            if len(child[c]) == 0 and len(parent[c]) == 1 and candidate1.isNotTerminal():
                for candidate2 in parent[b]:
                    (a,y,d) = candidate2.get()
                    if candidate2.isNotTerminal():
                        '''
                        We found a candidate to remove (a,y,b,x,c) down to (a,y,b)
                        '''
                        nrf = AMRtriple(a,Edge(y[0],'*',deriv.count),b)
                        nrt = [candidate2,candidate1]
                        new_amr = (deriv.amrs[0]).clone()
                        new_amr.replace_triple(nrt[0],nrf)
                        new_amr.remove_triple(nrt[1])
                        new_rule = TripleRule(nrf,nrt,'LL')
                        derivs.append(CanonicalDeriv.derive(deriv,new_amr,new_rule))
                        return derivs
        return derivs

    def applyCircle(self,deriv,derivs):
        '''
        A->B becomes A->B->B (circle)
        '''

        parent = defaultdict(set)

        triples = deriv.get_triples()
        for trip in triples:
            (a,b,c) = trip.get()
            parent[c].add(trip)
        
        for candidate1 in triples:
            (a,b,c) = candidate1.get()
            if a == c and candidate1.isNotTerminal():
                for candidate2 in parent[c]:
                    (x,y,z) = candidate2.get()
                    if candidate2.isNotTerminal():
                        '''
                        We found a candidate to remove (x,y,a,b,a) down to (x,y,a)
                        '''
                        assert z == a, "Oh, noes: something funny happened here"
                        nrf = AMRtriple(x,Edge(y[0],'*',deriv.count),z)
                        nrt = [candidate2,candidate1]
                        new_amr = (deriv.amrs[0]).clone()
                        new_amr.replace_triple(nrt[0],nrf)
                        new_amr.remove_triple(nrt[1])
                        new_rule = TripleRule(nrf,nrt,'CC')
                        derivs.append(CanonicalDeriv.derive(deriv,new_amr,new_rule))
                        return derivs
        return derivs
