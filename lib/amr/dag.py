'''
Directed Acyclic Graphs.

@author: Daniel Bauer (dbauer)
@since: 2012-07-27
'''

#from rule import Rule

from collections import defaultdict
#from lib import pyparsing
from amr_parser import *
from operator import itemgetter
import functools
import unittest
import re
import sys
import copy

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

###UTILS###

class memoize(object):
    '''Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    From  http://wiki.python.org/moin/PythonDecoratorLibrary
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if args in self.cache:
           return self.cache[args]
        else:
           value = self.func(*args)
           self.cache[args] = value
           return value
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        if not type(obj) is Dag:
            return functools.partial(self.func, obj)
        return functools.partial(self.__call__, obj)

def flatten(ll):
  """
  Flatten a nested list.
  """
  if isinstance(ll, list):
    if len(ll) == 0:
      return []
    else:
      return flatten(ll[0]) + flatten(ll[1:])
  else: 
    return [ll]
  
###Decode parse AST
def ast_to_dag(ast):
    """
    Convert the abstract syntax tree returned by the dag parser into an dag.
    """
    dag = Dag()

    def rec_step(x):  # Closure over dag

        node, concept, roles = x         
        if type(node) is str:
            node = node.replace("@","")
            for role, child in roles:
                if type(child) == tuple and len(child) == 3:
                    childnode = child[0]                                           
                    if type(childnode) is str and childnode.startswith("@"):
                        childnode = childnode.replace("@","")
                        dag.external_nodes.append(childnode)
                    tuple_child = (childnode,)
                    dag[node].append(role, tuple_child)
                    x = dag[childnode]
                    rec_step(child)

                elif type(child) == list: #Hyperedge 
                    childnode = set()
                    for c in child: 
                        if type(c) == tuple and len(c) == 3:
                            if type(c[0]) is str and c[0].startswith("@"):
                                new_c = c[0].replace("@","")
                                dag.external_nodes.append(new_c)
                            else: 
                                new_c = c[0]
                            childnode.add(new_c)
                            x = dag[new_c]
                            rec_step(c)
                        else:
                            if type(c) is str and c.startswith("@"):
                                c = c.replace("@","")
                                dag.external_nodes.append(c)
                            x = dag[c]
                            childnode.add(c)
                    newchild = tuple(childnode)        
                    dag[node].append(role, newchild)

                else: # Just assume this node is some special symbol
                    if type(child) is str and child.startswith("@"):
                        child = child.replace("@","")
                        tuple_child = (child,)
                        dag.external_nodes.append(child)
                        dag[node].append(role, tuple_child)
                        x = dag[child]
                    else:
                        dag[node].append(role, (child,))
                        x = dag[child]


    root = ast[0]
    if type(root) == tuple and len(root) == 3: 
        if "@" in root[0]:
            dag.external_nodes.append(root[0].replace("@",""))  

        dag.roots.append(root[0].replace("@",""))
       
        rec_step(root)
    else: 
        if "@" in root:
            dag.external_nodes.append(root.replace("@",""))        
        dag.roots.append(root.replace("@",""))

    return dag 

###

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


class NonterminalLabel(object):
    """
    There can be multiple nonterminal edges with the same symbol. Wrap the 
    edge into an object so two edges do not compare equal.
    Nonterminal edges carry a nonterminal symbol and an index that identifies
    it uniquely in a rule.
    """

    label_matcher = re.compile("(?P<label>.*?)(\[(?P<index>.*)\])?$")

    def __init__(self, label, index = None):            
        if index is not None:
            self.label = label
            self.index = index  
        else: 
            match = NonterminalLabel.label_matcher.match(label)
            self.index = match.group("index")
            self.label = match.group("label")

    def __eq__(self, other):
        try: 
            return self.label == other.label and self.index == other.index
        except AttributeError:
            return False     
    
    def __repr__(self):
        return "NT(%s)" % str(self)

    def __str__(self):
        if self.index is not None:
            return "#%s[%s]" % (str(self.label), str(self.index))
        else: 
            return "#%s" % str(self.label)

    def __hash__(self):
        return 83 * hash(self.label) + 17 * hash(self.index)



class Dag(defaultdict):
    """
    A directed acyclic graph permitting duplicate outgoing edge labels.
    """
   
    _parser_singleton = None # Store singleton AMR/DAG parser as a class variable

    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, ListMap, *args, **kwargs) 
        self.roots = []
        self.external_nodes = [] 
        self.replace_count = 0    # Count how many replacements have occured in this DAG
                                  # to prefix unique new node IDs for glued fragments.

        self.__cached_triples = None
        self.__cached_depth = None

        self.node_alignments = {} 
        self.edge_alignments = {}

    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + (self.__dict__,) +t[3:]
        
    @classmethod
    def from_string(cls, s):
        """
        Initialize a new DAG from a Pennman style string.
        """
        #if not cls._parser_singleton: # Initialize the AMR parser only once
        #    _parser_singleton = make_amr_parser()           
        #try:
        #    ast = _parser_singleton.parseString(s)
        #except pyparsing.ParseException, e:
        #    sys.stderr.write("Could not parse AMR: %s" % s)
        #    raise e 
        #return ast_to_dag(ast)

        if not cls._parser_singleton: # Initialize the AMR parser only once
            from graph_description_parser import GraphDescriptionParser, LexerError, ParserError
            _parser_singleton = GraphDescriptionParser() 
        try:
            amr = _parser_singleton.parse_string(s, concepts = False)
            return amr
        except (ParserError, LexerError), e:
            #sys.stderr.write("Could not parse DAG: %s" % s)
            raise e 

    @classmethod
    def from_triples(self, triples, roots=None, warn=sys.stderr):    
        """
        Initialize a new DAG from a list of (parent, relation, child) triples.
        Optionally pass a list of root nodes (if empty, roots will be determined
        automatically).

        >>> y = Dag.from_triples([('3', u'ARG1', ('5',)), ('2', u'ARG1', ('4',)), ('2', 'location', ('0',)), ('0', 'name', ('1',)), ('1', 'op4', (u'"Exchange"',)), ('1', 'op1', (u'"New"',)), ('1', 'op2', (u'"York"',)), ('1', 'op3', (u'"Stock"',))])               
        >>> x = Dag.from_triples(y.triples() + [('4', 'mod', '2')]) #doctest:+ELLIPSIS

        """
        dag = Dag() # Make new DAG

        for parent, relation, child in triples: 
            if isinstance(parent, basestring):
                new_par = parent.replace("@","")       
                if parent.startswith("@"):
                    dag.external_nodes.append(new_par)
            else:
                new_par = parent

            if type(child) is tuple: 
                new_child = []
                for c in child: 
                    if isinstance(c, basestring):
                        new_c = c.replace("@","")
                        new_child.append(new_c)
                        nothing = dag[new_c]
                        if c.startswith("@"):
                            dag.external_nodes.append(new_c)
                    else:
                        nothing = dag[c] 
                        new_child.append(c)
                new_child = tuple(new_child)
            else: # Allow triples to have single string children for convenience. 
                  # and downward compatibility.
                if isinstance(child, basestring):
                    tmpchild = child.replace("@","")
                    if child.startswith("@"):
                        dag.external_nodes.append(tmpchild)
                    new_child = (tmpchild,)
                    nothing = dag[tmpchild]
                else:
                    new_child = (child,)
                    nothing = dag[child]
            
            dag._add_triple(new_par, relation, new_child, warn=warn)
        
        # Allow the passed root to be either an iterable of roots or a single root
        if roots: 
            try:  # Try to interpret roots as iterable
                dag.roots.extend(roots)
            except TypeError: # If this fails just use the whole object as root
                dag.roots = list([roots])
        else: 
            dag.roots = dag.find_roots(warn=warn)        
        return dag



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

    def clone(self):
        """
        Return a deep copy of the DAG.
        """
        new = Dag()
        new.roots = copy.copy(self.roots)
        new.external_nodes = self.external_nodes
        new.node_alignments, new.edge_alignments = self.node_alignments, self.edge_alignments
        for tr in self.triples():
            new._add_triple(*copy.copy(tr))
        return new
    
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
    
    ### Methods that provide information about this DAG in different formats.###
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
        return par in self and rel in self[par] and child in self[par].getall(rel) 
    
    def triples(self, start_node = None, refresh = False, **kwargs):
        """
        Traverse the DAG breadth first to collect a list of (parent, relation, child) triples.
        """

        if (not (refresh or start_node)) and self.__cached_triples:
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

        if not start_node:
            self.__cached_triples = triples
            self.__cached_depth = triple_to_depth
        return triples 

    def get_all_depths(self):
        if not self.__cached_depth:
            self.triples()
        return self.__cached_depth

    def get_depth(self, triple):
        if not self.__cached_depth:
            self.triples()
        return self.__cached_depth[triple]

    def out_edges(self, node): 
        """
        Return outgoing edges from this node.
        """
        assert node in self
        return [(node, rel, child) for rel, child in self[node].items()]        

    def root_edges(self):
        """
        Return a list of out_going edges from the root nodes.
        """
        return flatten([self.out_edges(r) for r in self.roots])

    def get_all_in_edges(self):
        """
        Return dictionary mapping nodes to their incomping edges. 
        """
        res = defaultdict(list)
        for node, rel, child in self.triples():
            if type(child) is tuple:
                for c in child:
                    res[c].append((node,rel,child))
            else:
                res[child].append((node,rel,child))
        return res

    def in_edges(self, node):
        """
        Return incoming edges for a single node.
        """
        return self.get_all_in_edges()[node]

    def has_triple(self, parent, relation, child):
        """
        Return True if the DAG contains the given triple.
        """
        try:
            result = child in self[parent].get(relation)   
        except (TypeError, ValueError):
            return False
        return result

    def nonterminal_edges(self):
        """
        Retrieve all nonterminal labels from the DAG.
        """
        return [t for t in self.triples() if isinstance(t[1], NonterminalLabel)]
    
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

    ####Methods that modify the DAG###    
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

    ###Specific methods for hyperedge replacement###
    def apply_node_map(self, node_map):
        """
        Needed to compute SMATCH scores. Apply a mapping between node IDs to the DAG. 
        Unmapped IDs are preserved.
        """
        new = Dag()
        new.roots = [node_map[x] if x in node_map else x for x in self.roots ]
        new.external_nodes = [node_map[x] if x in node_map else x for x in self.external_nodes]

        for node in self.node_alignments:
            new.node_alignments[node_map[node]] = self.node_alignments[node]
        for par, rel, child in self.edge_alignments:
            if type(child) is tuple: 
                new.edge_alignments[(node_map[par] if par in node_map else par, rel, tuple([(node_map[c] if c in node_map else c) for c in child]))] = self.edge_alignments[(par, rel, child)]
            else: 
                new.edge_alignments[(node_map[par] if par in node_map else par, rel, node_map[child] if child in node_map else child)] = self.edge_alignments[(par, rel, child)]

        for par, rel, child in Dag.triples(self):
            if type(child) is tuple: 
                new._add_triple(node_map[par] if par in node_map else par, rel, tuple([(node_map[c] if c in node_map else c) for c in child]))
            else: 
                new._add_triple(node_map[par] if par in node_map else par, rel, node_map[child] if child in node_map else child)    
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
    
    def clone_canonical(self, external_dict = {}, prefix = ""):
        """
        Return a version of the DAG where all nodes have been replaced with canonical IDs.
        """
        new = Dag()
        node_map = self._get_canonical_nodes(prefix)
        for k,v in external_dict.items():
                node_map[k] = v
        return self.apply_node_map(node_map)                        
        #new.roots = [node_map[x] for x in self.roots]
        #new.external_nodes = set([node_map[x] for x in self.external_nodes])
        #
        #for node in self.node_alignments:
        #    new.node_alignments[node_map[node]] = self.node_alignments[node]
        #
        #for par, rel, child in self.edge_alignments:
        #    if type(child) is tuple: 
        #        new.edge_alignments[(node_map[par], rel, tuple([node_map[c] for c in child]))] = self.edge_alignments[(par, rel, child)]
        #    else: 
        #        new.edge_alignments[(node_map[par] , rel, node_map[child])]
        #
        #for par, rel, child in self.triples(instances = False):
        #    if type(child) is tuple:                 
        #        new._add_triple(node_map[par], rel, tuple([node_map[c] for c in child]))
        #    else: 
        #        new._add_triple(node_map[par], rel, node_map[child])    
        #return new

    def remove_fragment(self, dag):
        """
        Remove a collection of hyperedges from the DAG.
        """
        res_dag = Dag.from_triples([edge for edge in self.triples() if not dag.has_edge(*edge)])
        res_dag.roots = [r for r in self.roots if r in res_dag]
        res_dag.external_nodes = [n for n in self.external_nodes if n in res_dag]
        return res_dag

    def replace_fragment(self, dag, new_dag, partial_boundary_map = {}):
        """
        Replace a collection of hyperedges in the DAG with another collection of edges. 
        """
        # First get a mapping of boundary nodes in the new fragment to 
        # boundary nodes in the fragment to be replaced
        leaves = dag.find_leaves()
        external = new_dag.get_external_nodes()
        assert len(external) == len(leaves)
        boundary_map = dict(zip(external, leaves))
        dagroots = dag.find_roots() if not dag.roots else dag.roots
        assert len(dagroots) == len(new_dag.roots)
        for i in range(len(dagroots)):
            boundary_map[new_dag.roots[i]] = dagroots[i]
        boundary_map.update(partial_boundary_map)

        # now remove the old fragment
        res_dag = self.remove_fragment(dag)
        res_dag.roots = [boundary_map[x] if x in boundary_map else x for x in self.roots]
        res_dag.external_nodes = [boundary_map[x] if x in boundary_map else x for x in self.external_nodes]

        # and add the remaining edges, fusing boundary nodes
        for par, rel, child in new_dag.triples(): 
            
            new_par = boundary_map[par] if par in boundary_map else par

            if type(child) is tuple: #Hyperedge case
                new_child = tuple([boundary_map[c] if c in boundary_map else c for c in child])
            else:            
                new_child = boundary_map[child] if child in boundary_map else child
            res_dag._add_triple(new_par, rel, new_child)

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


    def collapse_fragment(self, dag, label = None, unary = False):
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
            res_dag._add_triple(r, label, external)
       
        res_dag.roots = self.roots
        return res_dag
    
    def apply_herg_rule(self, symbol, amr): 
        """
        Apply a hyperedge replacement grammar rule to this DAG.
        """
    
        #Find hyperedge to replace 
        for p, r, c in self.nonterminal_edges(): 
            if r.label == symbol:
                par, rel, child = p, r, c
                break
        repl_fragment = Dag.from_triples([(par,rel,child)])
        repl_fragment.roots = [par]
        #Replace it with the new fragment               
        new_amr = amr.clone_canonical(prefix = str(self.replace_count))        
        res_dag = self.replace_fragment(repl_fragment, new_amr)
        # Keep track of prefix for new canonical node IDs               
        res_dag.replace_count = self.replace_count + 1
        return res_dag

    ###Methods for rendering/viewing DAGs###
    def _get_gv_graph(self, *args):
        """node
        Return a pygraphviz AGraph representing the DAG.
        """
        require_graphics()
        graph = pgv.AGraph(strict=False,directed=True)
        graph.node_attr.update(height=0.1, width=0.1, shape='none')
        graph.edge_attr.update(fontsize='9')
        for node, rel, child in self.triples():
           nodestr, childstr = str(node), str(child)
           graph.add_edge(nodestr, childstr, label="%s"%rel)
        return graph

    def render(self, *args):
        """
        Interactively view the graph. Requires xdot and pygraphviz.
        """
        require_graphics()
        dot = self.get_dot()
        window = xdot.DotWindow()
        window.set_dotcode(dot)
    
    def get_dot(self, *args):
        """
        Return a graphviz dot representation. Requires pygraphviz.
        """
        return self._get_gv_graph().to_string()
   
    def render_to_file(self, file_or_name, *args, **kwargs):
        """
        Save a graphical representation of the dag using pygraphviz' draw
        method. 
        """
        graph = self._get_gv_graph()
        graph.draw(file_or_name, prog="dot", *args, **kwargs)

    ### Methods for string and python representation ###

    def to_string(self, newline = True):
        """
        Return a string representation of this DAG, with or without line breaks and indentation.
        """
        # Define extractor, combiner and hyperedgecombiner for DFS.
        def extractor(node, firsthit, leaf):
            if node is None:
                    return "root"

            if type(node) is tuple or type(node) is list: 
                return ",".join("@%s" % (n) if n in self.external_nodes else n for n in node)
            else: 
                return ("@%s" % str(node)) if node in self.external_nodes else str(node)

        def combiner(nodestr, childmap, depth):
            if newline:                                    
                childstr = " ".join(["\n%s :%s %s" % (depth * "\t", rel, child) for rel, child in sorted(childmap.items())])            
            else: 
                childstr = " ".join([":%s %s" % (rel, child) for rel, child in sorted(childmap.items())])            
            return "(%s %s)" % (nodestr, childstr)
        
        def hedgecombiner(nodes):
                return ", ".join(nodes)

        if newline: 
            return "\n".join(self.dfs(extractor, combiner, hedgecombiner))
        else: 
            return " ".join(self.dfs(extractor, combiner, hedgecombiner))
    
    def __repr__(self):	
        return "DAG{ %s }" % self.to_string(newline = False)        

    def __str__(self):
        return self.to_string(newline = True)

  


    ####Methods to provide representations suitable for David's HERG parser###
    #def to_david_repr(self):
    # 
    #    reentrant = self.get_reentrant_nodes()
    #    node_to_uid = dict((u,v) for (v,u) in enumerate(reentrant))
    #    
    #    def extractor(node, firsthit, leaf):
    #        if node is None:
    #            return "instance"
    #
    #        if type(node) is tuple or type(node) is list: 
    #            all_res = []
    #            for n in node:                     
    #                if n in self.external_nodes:
    #                    res = "@instance"
    #                else:
    #                    res = "instance"
    #                if n in node_to_uid: 
    #                    if firsthit:
    #                        if leaf: 
    #                            res = "#%i=(%s)" % (node_to_uid[n], res)
    #                        else: 
    #                            res = "#%i=" % (node_to_uid[n],res)
    #                    else:
    #                        res = "#%i" % (node_to_uid[n],res)
    #                all_res.append(res)
    #            return " ".join(all_res)
    #        else: 
    #            if node in self.external_nodes: 
    #                res = "@instance"
    #            else: 
    #                res = "instance"
    #            if node in node_to_uid: 
    #
    #                if firsthit:
    #                    if leaf: 
    #                        res = "#%i=(%s)" % (node_to_uid[node], res)    
    #                    else: 
    #                        res = "#%i=" % node_to_uid[node]
    #                else:
    #                    res = "#%i" % (node_to_uid[node])
    #
    #            return res 
    #
    #    def combiner(nodestr, childmap, depth):
    #        childstr = " ".join([":%s %s" % (str(rel).replace("#",""), child) for rel, child in sorted(childmap.items())])            
    #         
    #        if nodestr.startswith("#") and nodestr.endswith("="):
    #            return "%s(instance %s)" % (nodestr, childstr)
    #        else: 
    #            return "(%s %s)" % (nodestr, childstr)
    #
    #    def hedgecombiner(nodes):
    #            return " ".join(nodes)
    #
    #    return " ".join(self.dfs(extractor, combiner, hedgecombiner))




#class TestDag(unittest.TestCase):
#
#    def setUp(self):
#        self.dag =  Dag.from_string("(A :foo (B :blubb (D :fee E) :back C) :bar C)")
#    
#    def test_collapse(self):
#        self.dag =  Dag.from_string("(A :foo (B :blubb (D :fee E) :back C) :bar C)")
#        d2 = Dag.from_string("(A :foo (B :blubb D))")
#        self.assertEqual(self.dag.find_external_nodes(d2), ["B","D"])
#
#        d_gold = Dag.from_string("(A :bar C :new (B :back C), (D :fee E))")
#        self.assertEqual(self.dag.collapse_fragment(d2, "new"), d_gold)

# Special node objects
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




if __name__ == '__main__':
    #unittest.main()
   
    import doctest
    doctest.testmod()
