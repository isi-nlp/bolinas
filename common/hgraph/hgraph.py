'''
Hypergraph representation. 

@author: Daniel Bauer (dbauer)
@author: Nathan Schneider (nschneid)
@since: 2012-06-18
'''

from collections import defaultdict
import unittest
import re
import sys
import copy
from operator import itemgetter
from common.cfg import NonterminalLabel
from common.exceptions import DerivationException

_graphics = False
def require_graphics():
    global _graphics
    if _graphics: return
    
    # Try to import modules to render DAGs
    global xdot
    import xdot
    
    global pgv
    import pygraphviz as pgv
    
    _graphics = True

def print_amr_error(amr_str, warn=sys.stderr):
    warn.write("Could not parse AMR.\n")
    warn.write(amr_str)    
    warn.write("\n")

def conv(s):
    if not s: 
        return "NONE"
    if isinstance(s, StrLiteral):
        return s
    elif s.startswith('"') and s.endswith('"'):
        return  s[1:-1]
    else: 
        return s

class ListMap(defaultdict):
    '''
    A  map that can contain several values for the same key.
    @author: Nathan Schneider (nschneid)
    @since: 2012-06-18

    >>> x = ListMap()
    >>> x.append('mykey', 3)
    >>> x.append('key2', 'val')
    >>> x.append('mykey', 8)
    >>> x
    defaultdict(<type 'list'>, {'key2': ['val'], 'mykey': [3, 8]})
    >>> x['mykey']
    3
    >>> x.getall('mykey')
    [3, 8]
    >>> x.items()
    [('key2', 'val'), ('mykey', 3), ('mykey', 8)]
    >>> x.itemsfor('mykey')
    [('mykey', 3), ('mykey', 8)]
    >>> x.replace('mykey', 0)
    >>> x
    defaultdict(<type 'list'>, {'key2': ['val'], 'mykey': [0]})
    '''
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, list, *args, **kwargs)
    
    def __setitem__(self, k, v):
        if k in self:
            raise KeyError('Cannot assign to ListMap entry; use replace() or append()')
        return defaultdict.__setitem__(self, k, v)
    
    def __getitem__(self, k):
        '''Returns the *first* list entry for the key.'''
        return dict.__getitem__(self, k)[0]

    def getall(self, k):
        return dict.__getitem__(self, k)
        
    def items(self):
        return [(k,v) for k,vv in defaultdict.items(self) for v in vv]
    
    def values(self):
        return [v for k,v in self.items()]
    
    def itemsfor(self, k):
        return [(k,v) for v in self.getall(k)]
    
    def replace(self, k, v):
        defaultdict.__setitem__(self, k, [v])
        
    def append(self, k, v):
        defaultdict.__getitem__(self, k).append(v) 
    
    def remove(self, k, v):
        defaultdict.__getitem__(self, k).remove(v)        
        if not dict.__getitem__(self,k):
            del self[k]

    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + t[2:]



# Actual AMR class

class Hgraph(defaultdict):
    """
    An abstract meaning representation.
    The structure consists of nested mappings from role names to fillers.
    Because a concept may have multiple roles with the same name, 
    a ListMap data structure holds a list of fillers for each role.
    A set of (concept, role, filler) triples can be extracted as well.
    """
    _parser_singleton = None

    def __init__(self, *args, **kwargs):       

        defaultdict.__init__(self, ListMap, *args, **kwargs) 
        self.roots = []
        self.external_nodes = {}
        self.rev_external_nodes = {}
        self.replace_count = 0    # Count how many replacements have occured in this DAG
                                  # to prefix unique new node IDs for glued fragments.

        self.__cached_triples = None
        self.__cached_depth = None
        self.__nodelabels = False

        self.node_alignments = {} 
        self.edge_alignments = {}


        self.__cached_triples = None
        self.node_to_concepts = {}

    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + (self.__dict__,) +t[3:]

    ####Hashing methods###                        

    def _get_node_hashes(self):
        tabu = set()
        queue = []
        node_to_id = defaultdict(int) 
        for x in sorted(self.roots):
            if type(x) is tuple: 
                for y in x: 
                    queue.append((y,0))
                    node_to_id[y] = 0
            else:          
                queue.append((x,0)) 
                node_to_id[x] = 0    
        while queue: 
            node, depth = queue.pop(0)            
            if not node in tabu:
                tabu.add(node)
                rels = tuple(sorted(self[node].keys()))
                node_to_id[node] += 13 * depth + hash(rels) 
    
                for rel in rels:
                    children = self[node].getall(rel)
                    for child in children:
                        if not child in node_to_id: 
                            if type(child) is tuple:
                                for c in child: 
                                    node_to_hash = 41 * depth
                                    queue.append((c, depth+1))
                            else:
                                node_to_hash = 41 * depth 
                                queue.append((child, depth+1))
        return node_to_id                    
  
    def __hash__(self):
        # We compute a hash for each node in the AMR and then sum up the hashes. 
        # Colisions are minimized because each node hash is offset according to its distance from
        # the root.
        node_to_id = self._get_node_hashes()
        return sum(node_to_id[node] for node in node_to_id)

    def __eq__(self, other):
        return hash(self) == hash(other)
 

    @classmethod
    def from_concept_edge_labels(cls, amr):
        """
        Create a new AMR from an AMR or a DAG in which concepts are pushed into incoming edges.
        """
        new_amr = amr.clone() 
        new_amr.roots = copy.copy(amr.roots)
        for par, rel, child in amr.triples():
           if type(rel) is str:   
                parts = rel.rsplit(":",1)    
                part2 = None
                if len(parts)==2:
                    part1, part2 = parts
                    if not (part1.lower().startswith("root")):
                        new_amr._replace_triple(par, rel, child, par, part1, child)
                    for c in child: 
                        new_amr.node_to_concepts[c] = part2
                        if (par,rel,child) in amr.edge_alignments:                            
                            if not c in new_amr.node_alignments: 
                                new_amr.node_alignments[c] = []
                            new_amr.node_alignments[c].extend(amr.edge_alignments[(par,rel,child)])
                if rel.lower().startswith("root"): 
                    new_amr.roots.remove(par)
                    new_amr._remove_triple(par, rel, child)
                    new_amr.roots = []
                    for c in child:
                        new_amr.roots.append(c)
                elif par in amr.roots and par not in new_amr.node_to_concepts:
                    new_amr.node_to_concepts[par] = None    
        new_amr.edge_alignments = {} 
        return new_amr

    def to_concept_edge_labels(self, warn=False):
        """"
        Return an new DAG with equivalent structure as this AMR (plus additional root-edge), in
        which concepts are pushed into incoming edges.
        """

        new_amr = self.clone(warn=warn)
        for par, rel, child in self.triples(instances = False):
            #new_rel = "%s:%s" % (rel, ":".join(self.node_to_concepts[c] for c in child if c in self.node_to_concepts))
            children = [conv(self.node_to_concepts[c]) if c in self.node_to_concepts and self.node_to_concepts[c] else conv(c) for c in child]
            new_rel = '%s:%s' % (rel, ':'.join(children))
            new_amr._replace_triple(par,rel,child, par, new_rel, child, warn=warn)

            # Copy edge alignemnts
            if (par, rel, child) in self.edge_alignments:
                new_amr.edge_alignments[(par, new_rel, child)] = self.edge_alignments[(par,rel,child)]           
            # Copy node alignments of children    
            for c in child: 
                if c in self.node_alignments:
                    if not (par, new_rel, child) in new_amr.edge_alignments:
                       new_amr.edge_alignments[(par, new_rel, child)] = []
                    new_amr.edge_alignments[(par, new_rel, child)].extend(self.node_alignments[c])
            for e in new_amr.edge_alignments: 
                new_amr.edge_alignments[e] = list(set(new_amr.edge_alignments[e]))

        for r in self.roots:
            if r in self.node_to_concepts:
                new_rel = "ROOT:%s" % conv(self.node_to_concepts[r])
            else: 
                new_rel = "ROOT"
            newtriple =  ('root0', new_rel, (r,))
            new_amr._add_triple(*newtriple, warn=warn)
            new_amr.roots.remove(r)
            if not "root0" in new_amr.roots:
                new_amr.roots.append('root0' )

            if r in self.node_alignments:
                new_amr.edge_alignments[newtriple] = self.node_alignments[r]

        return new_amr

#    def make_rooted_amr(self, root, swap_callback=None and (lambda oldtrip,newtrip: True), warn=sys.stderr):
#        """
#        Flip edges in the AMR so that all nodes are reachable from the unique root.
#        If 'swap_callback' is provided, it is called whenever an edge is inverted with 
#        two arguments: the old triple and the new triple. 
#        >>> x =Hgraph.from_triples( [(u'j', u'ARG0', (u'p',)), (u'j', u'ARG1', (u'b',)), (u'j', u'ARGM-PRD', ('t',)), (u'j', 'time', ('d',)), (u'p', 'age', ('t1',)), (u'p', 'name', ('n',)), ('t', u'ARG0-of', ('d1',)), ('d', 'day', (29,)), ('d', 'month', (11,)), ('t1', 'quant', (61,)), ('t1', 'unit', ('y',)), ('n', 'op1', (u'"Pierre"',)), ('n', 'op2', (u'"Vinken"',)), ('d1', u'ARG0', ('t',)), ('d1', u'ARG3', (u'n1',))] , {u'b': u'board', 'd': 'date-entity', u'j': u'join-01-ROOT', 't1': 'temporal-quantity', u'p': u'person', 't': 'thing', 'y': 'year', u'n1': u'nonexecutive', 'n': 'name', 'd1': 'direct-01'} )
#        >>> x
#        DAG{ (j / join-01-ROOT :ARG0 (p / person :age (t1 / temporal-quantity :quant 61 :unit (y / year) ) :name (n / name :op1 "Pierre" :op2 "Vinken")) :ARG1 (b / board) :ARGM-PRD (t / thing :ARG0-of (d1 / direct-01 :ARG0 t :ARG3 (n1 / nonexecutive) )) :time (d / date-entity :day 29 :month 11)) }
#        >>> x.make_rooted_amr("n")
#        DAG{ (n / name :name-of (p / person :ARG0-of (j / join-01-ROOT :ARG1 (b / board) :ARGM-PRD (t / thing :ARG0-of (d1 / direct-01 :ARG0 t :ARG3 (n1 / nonexecutive) )) :time (d / date-entity :day 29 :month 11)) :age (t1 / temporal-quantity :quant 61 :unit (y / year) )) :op1 "Pierre" :op2 "Vinken") }
#        """
#        if not root in self:
#            raise ValueError, "%s is not a node in this AMR." % root    
#        amr = self.clone(warn=warn)
#
#        all_nodes = set(amr.get_nodes())
#
#        unreached  = True
#        while unreached: 
#            reach_triples = amr.triples(start_node = root, instances = False)
#            reached = set()
#            reached.add(root)
#            for p,r,c in reach_triples: 
#                reached.add(p)
#                reached.update(c)
#
#            unreached = all_nodes - reached
#     
#            out_triples = [(p,r,c) for p,r,c in amr.triples(refresh = True, instances = False) if c[0] in reached and p in unreached]
#            for p,r,c in out_triples:
#                newtrip = (c[0],"%s-of" %r, (p,))
#                amr._replace_triple(p,r,c,*newtrip, warn=warn)
#                if swap_callback: swap_callback((p,r,c),newtrip)
#        amr.triples(refresh = True)            
#        amr.roots = [root]
#        amr.node_alignments = self.node_alignments
#        return amr    

    def stringify(self, warn=False):
        """
        Convert all special symbols in the AMR to strings.
        """
        
        new_amr = Hgraph()

        def conv(node): # Closure over new_amr
            if isinstance(node, StrLiteral):
                var =  str(node)[1:-1] 
                new_amr._set_concept(var, str(node))
                return var
            else: 
                return str(node)

        for p,r,c in self.triples(instances = False):
            c_new = tuple([conv(child) for child in c]) if type(c) is tuple else conv(c)
            p_new = conv(p)
            new_amr._add_triple(p_new, r, c_new, warn=warn)

        new_amr.roots = [conv(r) for r in self.roots]
        new_amr.external_nodes = dict((conv(r),val) for r,val in self.external_nodes.items())
        new_amr.rev_external_nodes = dict((val, conv(r)) for val,r in self.rev_external_nodes.items())
        new_amr.edge_alignments = self.edge_alignments
        new_amr.node_alignments = self.node_alignments
        for node in self.node_to_concepts:    
            new_amr._set_concept(conv(node), self.node_to_concepts[node])
        return new_amr    

    # Class methods to create new AMRs
    @classmethod
    def from_string(cls, amr_string):
        """
        Initialize a new abstract meaning representation from a Pennman style string.
        """
        if not cls._parser_singleton: # Initialize the AMR parser only once
            from graph_description_parser import GraphDescriptionParser, LexerError, ParserError 
            _parser_singleton = GraphDescriptionParser() 
            amr = _parser_singleton.parse_string(amr_string)
            return amr

    @classmethod
    def from_triples(cls, triples, concepts, roots=None, warn=sys.stderr):
        """
        Initialize a new hypergraph from a collection of triples and a node to concept map.
        """
        
        graph = Hgraph() # Make new DAG

        for parent, relation, child in triples: 
            if isinstance(parent, basestring):
                new_par = parent.replace("@","")       
                if parent.startswith("@"):
                    graph.external_nodes.append(new_par)
            else:
                new_par = parent

            if type(child) is tuple: 
                new_child = []
                for c in child: 
                    if isinstance(c, basestring):
                        new_c = c.replace("@","")
                        new_child.append(new_c)
                        nothing = graph[new_c]
                        if c.startswith("@"):
                            graph.external_nodes.append(new_c)
                    else:
                        nothing = graph[c] 
                        new_child.append(c)
                new_child = tuple(new_child)
            else: # Allow triples to have single string children for convenience. 
                  # and downward compatibility.
                if isinstance(child, basestring):
                    tmpchild = child.replace("@","")
                    if child.startswith("@"):
                        graph.external_nodes.append(tmpchild)
                    new_child = (tmpchild,)
                    nothing = graph[tmpchild]
                else:
                    new_child = (child,)
                    nothing = graph[child]
            
            graph._add_triple(new_par, relation, new_child, warn=warn)
        
        # Allow the passed root to be either an iterable of roots or a single root
        if roots: 
            try:  # Try to interpret roots as iterable
                graph.roots.extend(roots)
            except TypeError: # If this fails just use the whole object as root
                graph.roots = list([roots])
        else: 
            graph.roots = graph.find_roots(warn=warn)        

        graph.node_to_concepts = concepts
        graph.__cached_triples = None
        return graph

    # Methods that create new AMRs


    def get_concept(self, node):
        """
        Retrieve the concept name for a node.
        """
        return self.node_to_concepts[node]
    
    def _set_concept(self, node, concept):
        """
        Set concept name for a node.
        """
        self.node_to_concepts[node] = concept


    def get_nodes(self):
        """
        Return the set of node identifiers in the DAG.
        """
        # The default dictionary creates keys for hyperedges... not sure why.
        # We just ignore them.
        ordered = self.get_ordered_nodes()
        res = ordered.keys()
        res.sort(lambda x,y: cmp(ordered[x], ordered[y]))
        return res

    def has_edge(self, par, rel, child):
        return self.has_triple(par, rel, child)

 
    def has_triple(self, parent, relation, child):
        """
        Return True if the DAG contains the given triple.
        """
        try:
            result = child in self[parent].get(relation)   
        except (TypeError, ValueError):
            return False
        return result

    def get_all_depths(self):
        if not self.__cached_depth:
            self.triples()
        return self.__cached_depth

    def get_depth(self, triple):
        if not self.__cached_depth:
            self.triples()
        return self.__cached_depth[triple]

    def out_edges(self, node, nodelabels = False): 
        """
        Return outgoing edges from this node.
        """
        assert node in self
        if nodelabels: 
            result = []
            nlabel = self.node_to_concepts[node]       
            for rel, child in self[node].items():
                if type(child) is tuple:
                    nchild = tuple([(c, self.node_to_concepts[c]) for c in child])
                else:
                    nchild = (child, self.node_to_concepts[child])
                result.append(((node, nlabel), rel, nchild))
            return result

        return [(node, rel, child) for rel, child in self[node].items()]        

    #def root_edges(self):
    #    """
    #    Return a list of out_going edges from the root nodes.
    #    """
    #    return flatten([self.out_edges(r) for r in self.roots])

    def get_all_in_edges(self, nodelabels = False):
        """
        Return dictionary mapping nodes to their incomping edges. 
        """
        res = defaultdict(list)
        for node, rel, child in self.triples(nodelabels = nodelabels):
            if type(child) is tuple:
                for c in child:
                    if nodelabels: 
                        res[c[0]].append((node,rel,child))
                    else: 
                        res[c].append((node,rel,child))
            else:
                if nodelabels: 
                    res[child].append((node,rel,child))
                else:
                    res[child[0]].append((node,rel,child))
        return res

    def in_edges(self, node, nodelabels = False):
        """
        Return incoming edges for a single node.
        """
        return self.get_all_in_edges(nodelabels)[node]

    def nonterminal_edges(self):
        """
        Retrieve all nonterminal labels from the DAG.
        """
        return [t for t in self.triples() if isinstance(t[1], NonterminalLabel)]

    def get_terminals_and_nonterminals(self, nodelabels = False):
        """
        Return a tuple in which the first element is a set of all terminal labels
        and the second element is a set of all nonterminal labels.
        """
        # This is used to compute reachability of grammar rules
        terminals = set()
        nonterminals = set()
        for p,r,children in self.triples():
            if isinstance(r, NonterminalLabel):
                nonterminals.add(r.label)
            else:
                if nodelabels: 
                    terminals.add((self.node_to_concepts[p],r,tuple([self.node_to_concepts[c] for c in children])))
                else: 
                    terminals.add(r)
        return terminals, nonterminals

    def get_external_nodes(self):
        """
        Retrieve the list of external nodes of this dag fragment.
        """
        return self.external_nodes 
    
    def reach(self, node):
        """
        Return the set of nodes reachable from a node
        """
        res = set()
        for p,r,c in self.triples(start_node = node, instances = False):
            res.add(p)
            if type(c) is tuple:
                res.update(c)
            else:
                res.add(c)
        return res
   
    def find_roots(self, warn=sys.stderr):
        """
        Find and return a set of the roots of the DAG. This does NOT set the 'roots' attribute.
        """
        # there cannot be an odering of root nodes so it is okay to return a set
        parents = set()
        for k in self.keys():
            if type(k) is tuple:
                parents.update(k)
            else: 
                parents.add(k)
        children = set()
        for node in parents: 
            for v in self[node].values():
                if type(v) is tuple: 
                    children.update(v)
                else:
                    children.add(v)
        roots = list(parents - children)

        not_found = parents.union(children)
        for r in roots: 
            x = self.triples(start_node = r, instances = False)
            for p,r,c in x: 
               if p in not_found:
                   not_found.remove(p)
               if type(c) is tuple: 
                   for ch in c: 
                       if ch in not_found:
                           not_found.remove(ch)
               if c in not_found: 
                   not_found.remove(c)

        while not_found:
            parents = sorted([x for x in not_found if self[x]], key=lambda a:len(self.triples(start_node = a)))
            if not parents: 
                if warn: warn.write("WARNING: orphaned leafs %s.\n" % str(not_found))
                roots.extend(list(not_found))
                return roots
            new_root = parents.pop()
            for p,r,c in  self.triples(start_node = new_root):
               if p in not_found:
                   not_found.remove(p)
               if type(c) is tuple: 
                   for ch in c: 
                       if ch in not_found:
                           not_found.remove(ch)
               if c in not_found: 
                   not_found.remove(c)
            roots.append(new_root)    
        return roots    
  
    def get_ordered_nodes(self):
        """
        Get an mapping of nodes in this DAG to integers specifying a total order of 
        nodes. (partial order broken according to edge_label).
        """
        order = {}
        count = 0
        for par, rel, child in self.triples(instances = False):
            if not par in order: 
                order[par] = count 
                count += 1
            if type(child) is tuple:
                for c in child: 
                    if not c in order:
                        order[c] = count
                        count += 1
            else:        
                if not child in order: 
                    order[child] = count
                    count += 1
        return order

    def find_leaves(self):
        """
        Get all leaves in a DAG.
        """
        out_count = defaultdict(int)
        for par, rel, child in self.triples():
            out_count[par] += 1
            if type(child) is tuple:
               for c in child: 
                if not c in out_count: 
                    out_count[c] = 0
            else:
                if not child in out_count: 
                    out_count[child] = 0
        result = [n for n in out_count if out_count[n]==0]
        order = self.get_ordered_nodes()
        result.sort(lambda x,y: cmp(order[x], order[y]))
        return result

    def get_reentrant_nodes(self):
        """
        Get a list of nodes that have an in-degree > 1.
        """
        in_count = defaultdict(int)
        for par, rel, child in self.triples():
            if type(child) is tuple:
                for c in child: 
                    in_count[c] += 1
            else:
                in_count[child] += 1
        result = [n for n in in_count if in_count[n]>1] 
        order = self.get_ordered_nodes()
        result.sort(lambda x,y: cmp(order[x], order[y]))
        return result
   
    def get_weakly_connected_roots(self, warn=sys.stderr):
        """
        Return a set of root nodes for each weakly connected component.
        >>> x = Dag.from_triples([("a","B","c"), ("d","E","f")])
        >>> x.get_weakly_connected_roots()
        set(['a', 'd'])
        >>> y = Dag.from_triples([("a","B","c"), ("d","E","f"),("c","H","f")],{})
        >>> y.get_weakly_connected_roots()
        set(['a'])
        >>> y.is_connected()
        True
        """

        roots = list(self.find_roots(warn=warn))
        if len(roots) == 1:
                return roots

        merged  = defaultdict(list)
        node_to_comp = defaultdict(list)
        equiv = {}
        for r in roots:
            for n in self.reach(r):
                node_to_comp[n].append(r)
                if len(node_to_comp[n]) == 2: 
                    if not r in equiv:
                        equiv[r] = node_to_comp[n][0]    
                
        final = set()
        for r in roots: 
            unique_repr = r
            while unique_repr in equiv:
                unique_repr = equiv[unique_repr]        
            final.add(unique_repr)
              
        return final
        #new_roots = set()                       
        #for r in nodes:

    def is_connected(self, warn=sys.stderr):
        return len(self.get_weakly_connected_roots(warn=warn)) == 1        


 

    # Methods that traverse the hypergraph and represent it in different ways
    def dfs(self, extractor = lambda node, firsthit, leaf: node.__repr__(), combiner = lambda par,\
            childmap, depth: {par: childmap.items()}, hedge_combiner = lambda x: tuple(x)):
        """
        Recursively traverse the dag depth first starting at node. When traveling up through the
        recursion a value is extracted from each child node using the provided extractor method,
        then the values are combined using the provided combiner method. At the root node the
        result of the combiner is returned. Extractor takes a "firsthit" argument that is true
        the first time a node is touched. 
        """
        tabu = set()
        tabu_edge = set()

        def rec_step(node, depth):
            if type(node) is tuple: # Hyperedge case
                pass
            else:                
                node = (node,)
            allnodes = []
            for n in node: 
                firsthit = not n in tabu
                tabu.add(n)
                leaf = False if self[n] else True
                #firsthit = not node in tabu
                extracted = extractor(n, firsthit, leaf)
                child_map = ListMap()
                for rel, child in self[n].items():
                    if not (n, rel, child) in tabu_edge:
                        if child in tabu:
                            child_map.append(rel, extractor(child, False, leaf))
                            #pass
                        else:
                            tabu_edge.add((n, rel, child))
                            child_map.append(rel, rec_step(child, depth + 1))
                if child_map: 
                    combined = combiner(extracted, child_map, depth)
                    allnodes.append(combined)
                else: 
                    allnodes.append(extracted)
            return hedge_combiner(allnodes)

        return [rec_step(node, 0) for node in self.roots]

    def triples(self, instances =  False, start_node = None, refresh = False, nodelabels = False):
        """
        Retrieve a list of (parent, edge-label, tails) triples. 
        """

        if (not (refresh or start_node or nodelabels!=self.__nodelabels)) and self.__cached_triples:
            return self.__cached_triples

        triple_to_depth = {}
        triples = []
        tabu = set()

        if start_node:
            queue = [(start_node,0)]
        else:             
            queue = [(x,0) for x in self.roots]
        while queue: 
            node, depth = queue.pop(0)
            if not node in tabu:
                tabu.add(node)
                for rel, child in sorted(self[node].items(), key=itemgetter(0)):
                    if nodelabels:
                        newchild = tuple([(n,self.node_to_concepts[n]) for n in child])
                        newnode = (node, self.node_to_concepts[node])
                        t = (newnode, rel, newchild)
                    else:
                        t = (node, rel, child)

                    triples.append(t) 
                    triple_to_depth[t] = depth
                    if type(child) is tuple:                            
                        for c in child:
                            if not c in tabu:
                                queue.append((c, depth+1))
                    else:
                        if not child in tabu: 
                            queue.append((child, depth+1))

        #if  instances: 
        #    if instances:
        #        for node, concept in self.node_to_concepts.items():
        #            triples.append((node, 'instance', concept))
        #    self.__cached_triples = res                        
        
        if not start_node:
            self.__cached_triples = triples
            self.__cached_depth = triple_to_depth
            self.__nodelabels = nodelabels

        return triples 

    def __str__(self):

        reentrant_nodes = self.get_reentrant_nodes()

        def extractor(node, firsthit, leaf):
            if node is None:
                    return "root"
            if type(node) is tuple or type(node) is list: 
                return " ".join("%s*%i" % (n, self.external_nodes[n]) if n in self.external_nodes else n for n in node)
            else:
                if type(node) is int or type(node) is float or isinstance(node, (Literal, StrLiteral)):
                    return str(node)
                else: 
                    if firsthit:
                        if node in self.node_to_concepts and self.node_to_concepts[node]: 
                            concept = self.node_to_concepts[node]
                            if node in self.external_nodes:    
                                return "%s%s*%i " % ("%s." % node if node in reentrant_nodes else "", concept, self.external_nodes[node])
                            else:
                                return "%s%s " % ("%s." % node if node in reentrant_nodes else "", concept)
                        else:
                            if node in self.external_nodes:    
                                return "%s.*%i " % (node if node in reentrant_nodes else "", self.external_nodes[node])
                            else:
                                return "%s." % (node if node in reentrant_nodes else "")
                    else:
                        return "%s." % (node if node in reentrant_nodes else "") 

        def combiner(nodestr, childmap, depth):
            childstr = " ".join(["\n%s %s %s" % (depth * "\t", ":%s" % rel if rel else "", child) for rel, child in sorted(childmap.items())])            
            return "(%s %s)" % (nodestr, childstr)

        def hedgecombiner(nodes):
             return " ".join(nodes)

        return " ".join(self.dfs(extractor, combiner, hedgecombiner))
   

    def to_amr_string(self):

        def extractor(node, firsthit, leaf):
            if node is None:
                    return "root"
            if type(node) is tuple or type(node) is list:
                return ",".join("@%s" % (n) if n in self.external_nodes else n for n in node)
            else:
                if type(node) is int or type(node) is float or isinstance(node, (Literal, StrLiteral)):
                    alignmentstr = "~e.%s" % ",".join(str(x) for x in self.node_alignments[node]) if node in self.node_alignments else ""
                    return "%s%s" % (str(node), alignmentstr)
                else:
                    if firsthit and node in self.node_to_concepts:
                        concept = self.node_to_concepts[node]
                        alignmentstr = "~e.%s" % ",".join(str(x) for x in self.node_alignments[node]) if node in self.node_alignments else ""
                        if not self[node]:
                            if node in self.external_nodes:
                                return "(@%s / %s%s) " % (node, concept, alignmentstr)
                            else:
                                return "(%s / %s%s) " % (node, concept, alignmentstr)
                        else:
                            if node in self.external_nodes:
                                return "@%s / %s%s " % (node, concept, alignmentstr)
                            else:
                                return "%s / %s%s " % (node, concept, alignmentstr)
                    else:
                        if node in self.external_nodes:
                            return "@%s" % node
                        else:
                            return "%s" % node


        def combiner(nodestr, childmap, depth):
            childstr = " ".join(["\n%s :%s %s" % (depth * "\t", rel, child) for rel, child in sorted(childmap.items())])
            return "(%s %s)" % (nodestr, childstr)

        def hedgecombiner(nodes):
             return " ,".join(nodes)

        return "\n".join(self.dfs(extractor, combiner, hedgecombiner))
 
    def to_string(self, newline = False):
         if newline:
             return str(self)
         else:
             return re.sub("(\n|\s+)"," ",str(self))
    
    def graph_yield(self):
        """
        Return the yield of this graph (a list of all edge labels). 
        Hyperedge tentacles are ordered. If hyperedges are used to represent
        trees this returns the intuitive yield of this tree. 
        If a node has multiple children their order is abitrary.
        """
        tabu = set()

        def rec_step(node, depth):
            if type(node) is not tuple:
                node = (node,)
            allnodes = []
            for n in node: 
                firsthit = not n in tabu
                tabu.add(n)
                leaf = False if self[n] else True
                #firsthit = not node in tabu
                extracted = self.node_to_concepts[n] 
                #child_map = ListMap()
                if extracted: 
                    allnodes.append(extracted)
                for rel, child in self[n].items():           
                    if child in tabu:
                        allnodes.append(rel)
                    else:
                        if rel:
                            allnodes.append(rel)
                        if child: 
                            allnodes.extend(rec_step(child, depth +1))
            return allnodes

        return sum([rec_step(node, 0) for node in self.roots],[])
        

    def get_dot(self, instances = True):
        """
        Return a graphviz dot representation.
        """
        return self._get_gv_graph(instances).to_string()
    
    def _get_gv_graph(self, instances = True):
        """
        Return a pygraphviz AGraph.
        """
        require_graphics()
        graph = pgv.AGraph(strict=False,directed=True)
        graph.node_attr.update(height=0.1, width=0.1, shape='none')
        graph.edge_attr.update(fontsize='9')
        for node, rel, child in self.triples(instances):
           nodestr, childstr = node, child
           if not instances:
                if node in self.node_to_concepts: 
                    nodestr = "%s / %s" % (node, self.node_to_concepts[node])
                if child in self.node_to_concepts:
                    childstr = "%s / %s" % (child, self.node_to_concepts[child])
           graph.add_edge(nodestr, childstr, label=":%s"%rel)
        return graph
   
    def render(self, instances = True):
        """
        Interactively view the graph using xdot. 
        """
        require_graphics()
        dot = self.get_dot(instances)
        window = xdot.DotWindow()
        window.set_dotcode(dot)
    
    def render_to_file(self, file_or_name, instances = True, *args, **kwargs):
        """
        Save graph to file
        """
        graph = self._get_gv_graph(instances)
        graph.draw(file_or_name, prog="dot", *args, **kwargs)
    
    def clone(self, warn=sys.stderr):
        """
        Return a deep copy of the AMR.
        """
        new = Hgraph() 
        new.roots = copy.copy(self.roots)
        new.external_nodes = copy.copy(self.external_nodes)
        new.rev_external_nodes = copy.copy(self.rev_external_nodes)
        new.node_to_concepts = copy.copy(self.node_to_concepts)
        new.node_alignments, new.edge_alignments = self.node_alignments, self.edge_alignments
        for triple in self.triples(instances = False):
            new._add_triple(*copy.copy(triple), warn=warn)        
        return new

    def _get_canonical_nodes(self, prefix = ""):
        """
        Get a mapping from node identifiers to IDs of the form x[prefix]number.
        This uses the hash code for each node which only depend on DAG topology (not on node IDs).
        Therefore two DAGs with the same structure will receive the same canonical node labels.
        """
        # Get node hashes, then sort according to hash_code and use the index into this
        # sorted list as new ID.
        node_hashes = self._get_node_hashes()
        nodes = node_hashes.keys() 
        nodes.sort(lambda x,y: cmp(int(node_hashes[x]),int(node_hashes[y]))) 
        return dict([(node.replace("@",""), "x%s%s" % (prefix, str(node_id)) ) for node_id, node in enumerate(nodes)])


    def clone_canonical(self, external_dict = {}, prefix = "", warn=False):
        """
        Return a version of the DAG where all nodes have been replaced with canonical IDs.
        """
        new = Hgraph()
        node_map = self._get_canonical_nodes(prefix)
        for k,v in external_dict.items():
                node_map[k] = v
    
   
        #return self.apply_node_map(node_map)
   
        new.roots = [node_map[x] for x in self.roots]
        for node in self.node_alignments:
            new.node_alignments[node_map[node]] = self.node_alignments[node]
        for par, rel, child in self.edge_alignments:
            if type(child) is tuple: 
                new.edge_alignments[(node_map[par] if par in node_map else par, rel, tuple([(node_map[c] if c in node_map else c) for c in child]))] = self.edge_alignments[(par, rel, child)]
            else: 
                new.edge_alignments[(node_map[par] if par in node_map else par, rel, node_map[child] if child in node_map else child)] = self.edge_alignments[(par, rel, child)]
      
        new.external_nodes = dict((node_map[x], self.external_nodes[x]) for x in self.external_nodes)
        new.rev_external_nodes = dict((self.external_nodes[x], node_map[x]) for x in self.external_nodes)
        for par, rel, child in self.triples(instances = False):
            if type(child) is tuple:                 
                new._add_triple(node_map[par], rel, tuple([node_map[c] for c in child]), warn=warn)
            else: 
                new._add_triple(node_map[par], rel, node_map[child], warn=warn)    
        
        new.node_to_concepts = {}
        for node in self.node_to_concepts:
            if node in node_map:
                new.node_to_concepts[node_map[node]] = self.node_to_concepts[node]
            else: 
                new.node_to_concepts[node] = self.node_to_concepts[node]
        return new

    def apply_node_map(self, node_map, warn=False):
        new = Hgraph()
        new.roots = [node_map[x] if x in node_map else x for x in self.roots ]
        new.external_nodes = dict([(node_map[x], self.external_nodes[x]) if x in node_map else x for x in self.external_nodes])

        for node in self.node_alignments:
            new.node_alignments[node_map[node]] = self.node_alignments[node]
        for par, rel, child in self.edge_alignments:
            if type(child) is tuple: 
                new.edge_alignments[(node_map[par] if par in node_map else par, rel, tuple([(node_map[c] if c in node_map else c) for c in child]))] = self.edge_alignments[(par, rel, child)]
            else: 
                new.edge_alignments[(node_map[par] if par in node_map else par, rel, node_map[child] if child in node_map else child)] = self.edge_alignments[(par, rel, child)]

        for par, rel, child in Dag.triples(self):
            if type(child) is tuple: 
                new._add_triple(node_map[par] if par in node_map else par, rel, tuple([(node_map[c] if c in node_map else c) for c in child]), warn=warn)
            else: 
                new._add_triple(node_map[par] if par in node_map else par, rel, node_map[child] if child in node_map else child, warn=warn)    

        new.__cached_triples = None
        for n in self.node_to_concepts:
            if n in node_map:
                new.node_to_concepts[node_map[n]] = self.node_to_concepts[n]
            else:
               new.node_to_concepts[n] = self.node_to_concepts[n]
        return new
        
    def find_nt_edge(self, label, index):       
        for p,r,c in self.triples():
            if type(r) is NonterminalLabel:
                if r.label == label and r.index == index:
                    return p,r,c    

    def remove_fragment(self, dag):
        """
        Remove a collection of hyperedges from the DAG.
        """
        res_dag = Hgraph.from_triples([edge for edge in self.triples() if not dag.has_edge(*edge)], dag.node_to_concepts)
        res_dag.roots = [r for r in self.roots if r in res_dag]
        res_dag.external_nodes = dict([(n, self.external_nodes[n]) for n in self.external_nodes if n in res_dag])
        return res_dag

    def replace_fragment(self, dag, new_dag, partial_boundary_map = {}, warn=False):
        """
        Replace a collection of hyperedges in the DAG with another collection of edges. 
        """
        # First get a mapping of boundary nodes in the new fragment to 
        # boundary nodes in the fragment to be replaced
        leaves = dag.find_leaves()
        external = new_dag.get_external_nodes()
        assert len(external) == len(leaves)
        boundary_map = dict([(x, leaves[external[x]]) for x in external])
        dagroots = dag.find_roots() if not dag.roots else dag.roots
        assert len(dagroots) == len(new_dag.roots)
        for i in range(len(dagroots)):
            boundary_map[new_dag.roots[i]] = dagroots[i]
        boundary_map.update(partial_boundary_map)

        # Make sure node labels agree
        for x in boundary_map: 
            if new_dag.node_to_concepts[x] != dag.node_to_concepts[boundary_map[x]]:
                raise DerivationException, "Derivation produces contradictory node labels."         

        # now remove the old fragment
        res_dag = self.remove_fragment(dag)
        res_dag.roots = [boundary_map[x] if x in boundary_map else x for x in self.roots]
        res_dag.external_nodes = dict([(boundary_map[x], self.external_nodes[x]) if x in boundary_map else (x, self.external_nodes[x]) for x in self.external_nodes])

        # and add the remaining edges, fusing boundary nodes
        for par, rel, child in new_dag.triples(): 

            new_par = boundary_map[par] if par in boundary_map else par
            
            if type(child) is tuple: #Hyperedge case
                new_child = tuple([boundary_map[c] if c in boundary_map else c for c in child])
            else:            
                new_child = boundary_map[child] if child in boundary_map else child
            res_dag._add_triple(new_par, rel, new_child, warn=warn)
        res_dag.node_to_concepts.update(new_dag.node_to_concepts)
        return res_dag

    def find_external_nodes(self, dag):
        """
        Find the external nodes of the fragment dag in this Dag.
        """
        # All nodes in the fragment that have an edge which is not itself in the fragment
        dagroots = dag.roots if dag.roots else dag.find_roots()

        return [l for l in dag if self[l] and not l in dagroots and  \
                             (False in [dag.has_edge(*e) for e in self.in_edges(l)] or \
                             False in [dag.has_edge(*e) for e in self.out_edges(l)])]


    def collapse_fragment(self, dag, label = None, unary = False, warn=False):
        """
        Remove all edges in a collection and connect their boundary node with a single hyperedge.

        >>> d1 = Dag.from_string("(A :foo (B :blubb (D :fee E) :back C) :bar C)")
        >>> d2 = Dag.from_string("(A :foo (B :blubb D))")
        >>> d1.find_external_nodes(d2)
        ['B', 'D']
        >>> d_gold = Dag.from_string("(A :new (B :back C), (D :fee E) :bar C)")
        >>> d1.collapse_fragment(d2, "new") == d_gold
        True
        """

        dagroots = dag.find_roots() if not dag.roots else dag.roots

        if dag.external_nodes: # Can use specified external nodesnode
            external = tuple(set(self.find_external_nodes(dag) +
              dag.external_nodes))
        else:
            external = tuple(self.find_external_nodes(dag))
      
        if not unary and not external:
            # prevent unary edges if flag is set
            external = (dag.find_leaves()[0],)

        res_dag = self.remove_fragment(dag)

        for r in dagroots: 
            res_dag._add_triple(r, label, external, warn=warn)
       
        res_dag.roots = self.roots
        return res_dag
 

# Methods that change the hypergraph
    def _add_triple(self, parent, relation, child, warn=sys.stderr):
        """
        Add a (parent, relation, child) triple to the DAG. 
        """
        if type(child) is not tuple:
            child = (child,)
        if parent in child: 
            #raise Exception('self edge!')
            #sys.stderr.write("WARNING: Self-edge (%s, %s, %s).\n" % (parent, relation, child))
            if warn: warn.write("WARNING: Self-edge (%s, %s, %s).\n" % (parent, relation, child))
            #raise ValueError, "Cannot add self-edge (%s, %s, %s)." % (parent, relation, child)
        for c in child: 
            x = self[c]
            for rel, test in self[c].items():
                if parent in test: 
                   if warn: warn.write("WARNING: (%s, %s, %s) produces a cycle with (%s, %s, %s)\n" % (parent, relation, child, c, rel, test))
                    #raise ValueError,"(%s, %s, %s) would produce a cycle with (%s, %s, %s)" % (parent, relation, child, c, rel, test)
        self[parent].append(relation, child)    
    
    def _replace_triple(self, parent1, relation1, child1, parent2, relation2, child2, warn=sys.stderr):
        """
        Delete a (parent, relation, child) triple from the DAG. 
        """
        self._remove_triple(parent1, relation1, child1)
        self._add_triple(parent2, relation2, child2, warn=warn)
    
    def _remove_triple(self, parent, relation, child):
        """
        Delete a (parent, relation, child) triple from the DAG. 
        """
        try:
            self[parent].remove(relation, child)    
        except ValueError:
            raise ValueError, "(%s, %s, %s) is not an AMR edge." % (parent, relation, child) 


# HRG related methods    
    def compute_fw_table(self):
      table = dict()
      nodes = self.get_nodes()
      for node in nodes:
        table[(node,node)] = (0,None)
        for oedge in self.out_edges(node):
          for tnode in oedge[2]:
            table[(node,tnode)] = (1,oedge)
            table[(tnode,node)] = (1,oedge)
      for n_k in nodes:
        for n_i in nodes:
          for n_j in nodes:
            if not ((n_i, n_k) in table and (n_k, n_j) in table):
              continue
            k_dist = table[(n_i,n_k)][0] + table[(n_k,n_j)][0]
            k_edge_forward = table[(n_k,n_j)][1]
            k_edge_back = table[(n_k,n_i)][1]
            if (n_i, n_j) not in table or k_dist < table[(n_i,n_j)][0]:
              table[(n_i,n_j)] = (k_dist, k_edge_forward)
              table[(n_j,n_i)] = (k_dist, k_edge_back)

      self.fw_table = table

    def star(self, node):
      return frozenset(self.in_edges(node) + self.out_edges(node))


class StrLiteral(str):
    def __str__(self):
        return '"%s"' % "".join(self)

    def __repr__(self):
            return "".join(self)

class SpecialValue(str):
        pass

class Quantity(str):
        pass

class Literal(str):
    def __str__(self):
        return "'%s" % "".join(self)

    def __repr__(self):
            return "".join(self)




if __name__ == "__main__":

    import doctest
    doctest.testmod()



      
