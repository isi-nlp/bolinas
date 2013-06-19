'''
Abstract Meaning Representation

@author: Daniel Bauer (dbauer)
@author: Nathan Schneider (nschneid)
@since: 2012-06-18
'''

from dag import Dag, SpecialValue, Literal, StrLiteral, Quantity
from collections import defaultdict
from lib.cfg import NonterminalLabel
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

def print_amr_error(amr_str, warn=sys.stderr):
    warn.write("Could not parse AMR.\n")
    warn.write(amr_str)    
    warn.write("\n")


def conv(s):
    if not s: 
        return "NONE"
    if isinstance(s, StrLiteral):
        return s[1:-1] 
    elif s.startswith('"') and s.endswith('"'):
        return  s[1:-1]
    else: 
        return s


# Actual AMR class

class Amr(Dag):
    """
    An abstract meaning representation.
    The structure consists of nested mappings from role names to fillers.
    Because a concept may have multiple roles with the same name, 
    a ListMap data structure holds a list of fillers for each role.
    A set of (concept, role, filler) triples can be extracted as well.
    """
    _parser_singleton = None

    def __init__(self, *args, **kwargs):       
        super(Amr, self).__init__(*args, **kwargs)
        self.__cached_triples = None
        self.node_to_concepts = {}

    def apply_node_map(self, node_map):
        """
        Needed for smatch.
        """
        new = Dag.apply_node_map(self, node_map)
        new.__class__ = Amr
        new.node_to_concepts = {}
        new.__cached_triples = None
        for n in self.node_to_concepts:
            if n in node_map:
                new.node_to_concepts[node_map[n]] = self.node_to_concepts[n]
            else:
               new.node_to_concepts[n] = self.node_to_concepts[n]
        return new

    @classmethod
    def from_string(cls, amr_string):
        """
        Initialize a new abstract meaning representation from a Pennman style string.
        """
        #if not cls._parser_singleton: # Initialize the AMR parser only once
        #    _parser_singleton = make_amr_parser()           
        #try:
        #    ast = _parser_singleton.parseString(amr_string)
        #except pyparsing.ParseException, e:
        #    sys.stderr.write("Could not parse AMR: %s" % amr_string)
        #    raise e 
        #return ast_to_amr(ast)
        if not cls._parser_singleton: # Initialize the AMR parser only once
            from new_graph_description_parser import GraphDescriptionParser, LexerError, ParserError 
            _parser_singleton = GraphDescriptionParser() 
            amr = _parser_singleton.parse_string(amr_string)
            return amr

    @classmethod
    def from_concept_edge_labels(cls, amr):
        """
        Create a new AMR from an AMR or a DAG in which concepts are pushed into incoming edges.
        """
        if isinstance(amr, Amr):
            new_amr = amr.clone() 
        else:
            new_amr = Amr.from_triples(amr.triples(), {})
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
                if rel.lower().startswith("root"): 
                    new_amr.roots.remove(par)
                    new_amr._remove_triple(par, rel, child)
                    new_amr.roots = []
                    for c in child:
                        new_amr.roots.append(c)
                elif par in amr.roots and par not in new_amr.node_to_concepts:
                    new_amr.node_to_concepts[par] = None    
        #new_amr = amr.clone()
        #new_amr.__class__ = cls
        #new_amr.node_to_concepts = {}
        #for par, rel, child in amr.triples():
        #   if type(rel) is str:    
        #        part1, part2 = rel.rsplit(":",1)        
        #        if part2: 
        #            if part1.lower() != "root":
        #                new_amr._replace_triple(par, rel, child, par, part1,
        #                    child[0])
        #            new_amr.node_to_concepts[child[0]] = part2
        #        if part1.lower() == "root":
        #            new_amr.roots.remove(par)
        #            new_amr._remove_triple(par, rel, child)
        #            new_amr.roots.append(child[0])
        return new_amr

    def to_concept_edge_labels(self):
        """"
        Return an new DAG with equivalent structure as this AMR (plus additional root-edge), in
        which concepts are pushed into incoming edges.
        """

        new_amr = self.clone_as_dag()
        for par, rel, child in self.triples(instances = False):
            #new_rel = "%s:%s" % (rel, ":".join(self.node_to_concepts[c] for c in child if c in self.node_to_concepts))
            new_rel = '%s:%s' % (rel, ':'.join(conv(self.node_to_concepts[c]) if c in self.node_to_concepts else conv(c) for c in child))
            new_amr._replace_triple(par,rel,child, par, new_rel, child)

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
            new_amr._add_triple(*newtriple)
            new_amr.roots.remove(r)
            if not "root0" in new_amr.roots:
                new_amr.roots.append('root0' )

            if r in self.node_alignments:
                new_amr.edge_alignments[newtriple] = self.node_alignments[r]

        return new_amr

    def make_rooted_amr(self, root, swap_callback=None and (lambda oldtrip,newtrip: True), warn=sys.stderr):
        """
        Flip edges in the AMR so that all nodes are reachable from the unique root.
        If 'swap_callback' is provided, it is called whenever an edge is inverted with 
        two arguments: the old triple and the new triple. 
        >>> x =Amr.from_triples( [(u'j', u'ARG0', (u'p',)), (u'j', u'ARG1', (u'b',)), (u'j', u'ARGM-PRD', ('t',)), (u'j', 'time', ('d',)), (u'p', 'age', ('t1',)), (u'p', 'name', ('n',)), ('t', u'ARG0-of', ('d1',)), ('d', 'day', (29,)), ('d', 'month', (11,)), ('t1', 'quant', (61,)), ('t1', 'unit', ('y',)), ('n', 'op1', (u'"Pierre"',)), ('n', 'op2', (u'"Vinken"',)), ('d1', u'ARG0', ('t',)), ('d1', u'ARG3', (u'n1',))] , {u'b': u'board', 'd': 'date-entity', u'j': u'join-01-ROOT', 't1': 'temporal-quantity', u'p': u'person', 't': 'thing', 'y': 'year', u'n1': u'nonexecutive', 'n': 'name', 'd1': 'direct-01'} )
        >>> x
        DAG{ (j / join-01-ROOT :ARG0 (p / person :age (t1 / temporal-quantity :quant 61 :unit (y / year) ) :name (n / name :op1 "Pierre" :op2 "Vinken")) :ARG1 (b / board) :ARGM-PRD (t / thing :ARG0-of (d1 / direct-01 :ARG0 t :ARG3 (n1 / nonexecutive) )) :time (d / date-entity :day 29 :month 11)) }
        >>> x.make_rooted_amr("n")
        DAG{ (n / name :name-of (p / person :ARG0-of (j / join-01-ROOT :ARG1 (b / board) :ARGM-PRD (t / thing :ARG0-of (d1 / direct-01 :ARG0 t :ARG3 (n1 / nonexecutive) )) :time (d / date-entity :day 29 :month 11)) :age (t1 / temporal-quantity :quant 61 :unit (y / year) )) :op1 "Pierre" :op2 "Vinken") }
        """
        if not root in self:
            raise ValueError, "%s is not a node in this AMR." % root    
        amr = self.clone(warn=warn)

        all_nodes = set(amr.get_nodes())

        unreached  = True
        while unreached: 
            reach_triples = amr.triples(start_node = root, instances = False)
            reached = set()
            reached.add(root)
            for p,r,c in reach_triples: 
                reached.add(p)
                reached.update(c)

            unreached = all_nodes - reached
     
            out_triples = [(p,r,c) for p,r,c in amr.triples(refresh = True, instances = False) if c[0] in reached and p in unreached]
            for p,r,c in out_triples:
                newtrip = (c[0],"%s-of" %r, (p,))
                amr._replace_triple(p,r,c,*newtrip, warn=warn)
                if swap_callback: swap_callback((p,r,c),newtrip)
        amr.triples(refresh = True)            
        amr.roots = [root]
        amr.node_alignments = self.node_alignments
        return amr    

    def stringify(self):
        """
        Convert all special symbols in the AMR to strings.
        """
        
        new_amr = Amr()

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
            new_amr._add_triple(p_new, r, c_new)

        new_amr.roots = [conv(r) for r in self.roots]
        new_amr.external_nodes = dict((conv(r),val) for r,val in self.external_nodes.items())
        new_amr.rev_external_nodes = dict((val, conv(r)) for val,r in self.rev_external_nodes.items())
        new_amr.edge_alignments = self.edge_alignments
        new_amr.node_alignments = self.node_alignments
        for node in self.node_to_concepts:    
            new_amr._set_concept(conv(node), self.node_to_concepts[node])
        return new_amr    

    @classmethod
    def from_triples(cls, triples, concepts, roots=None, warn=sys.stderr):
        """
        Initialize a new abstract meaning representation from a collection of triples 
        and a node to concept map.
        """
        amr = Dag.from_triples(triples, roots, warn=warn)
        amr.__class__ = Amr
        amr.node_to_concepts = concepts
        amr.__cached_triples = None
        return amr

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

    def triples(self, instances =  False, start_node = None, refresh = False):
        """
        Retrieve a list of (node, role, filler) triples. If instances is False
        do not include 'instance' roles.
        """

        if not instances: 
            return super(Amr, self).triples(start_node, refresh)
        else: 
            if (not (refresh or start_node)) and self.__cached_triples:
                return self.__cached_triples
            
            res = copy.copy(super(Amr, self).triples(start_node, refresh))
            if instances:
                for node, concept in self.node_to_concepts.items():
                    res.append((node, 'instance', concept))
            self.__cached_triples = res                        
            return res

    def __str__(self):
        def extractor(node, firsthit, leaf):
            if node is None:
                    return "root"
            if type(node) is tuple or type(node) is list: 
                return " ".join("%s*%i" % (n, self.external_nodes[n]) if n in self.external_nodes else n for n in node)
            else: 
                if type(node) is int or type(node) is float or isinstance(node, (Literal, StrLiteral)):
                    return str(node)
                else: 
                    if firsthit and node in self.node_to_concepts and self.node_to_concepts[node]: 
                        concept = self.node_to_concepts[node]
                        if not self[node]:
                            if node in self.external_nodes:
                                return "%s.%s*%i " % (node, concept, self.external_nodes[node])
                            else:
                                return "%s.%s " % (node, concept)
                        else: 
                            if node in self.external_nodes:    
                                return "%s.%s*%i " % (node, concept, self.external_nodes[node])
                            else:
                                return "%s.%s " % (node, concept)
                    else:
                        return "%s." % node


        def combiner(nodestr, childmap, depth):
            childstr = " ".join(["\n%s :%s %s" % (depth * "\t", rel, child) for rel, child in sorted(childmap.items())])            
            return "(%s %s)" % (nodestr, childstr)

        def hedgecombiner(nodes):
             return " ".join(nodes)

        return " ".join(self.dfs(extractor, combiner, hedgecombiner))
    
    def to_string(self, newline = False):
         if newline:
             return str(self)
         else:
             return re.sub("(\n|\s+)"," ",str(self))

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
        Interactively view the graph. 
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
        new = Amr() 
        new.roots = copy.copy(self.roots)
        new.external_nodes = copy.copy(self.external_nodes)
        new.rev_external_nodes = copy.copy(self.rev_external_nodes)
        new.node_to_concepts = copy.copy(self.node_to_concepts)
        for triple in self.triples(instances = False):
            new._add_triple(*copy.copy(triple), warn=warn)        
        return new

    def clone_as_dag(self, instances = True):        
        """
        Return a copy of the AMR as DAG, ignoring node labels.
        """
        new = Dag()
        
        for triple in self.triples(instances = False): 
            new._add_triple(*copy.copy(triple))
        new.roots = copy.copy(self.roots)
        new.external_nodes = copy.copy(self.external_nodes)
        new.rev_external_nodes = copy.copy(self.rev_external_nodes)
        return new

    ### Dispatched to Dag
    def clone_canonical(self, external_dict = {}, prefix = ""):            
        """
        Return a version of the DAG where all nodes have been replaced with canonical IDs.
        """
        new = Amr()
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
                new._add_triple(node_map[par], rel, tuple([node_map[c] for c in child]))
            else: 
                new._add_triple(node_map[par], rel, node_map[child])    
        
        new.node_to_concepts = {}
        for node in self.node_to_concepts:
            if node in node_map:
                new.node_to_concepts[node_map[node]] = self.node_to_concepts[node]
            else: 
                new.node_to_concepts[node] = self.node_to_concepts[node]
        return new

    def apply_node_map(self, node_map, *args, **kwargs):
        new = Dag.apply_node_map(self, node_map, *args, **kwargs)    
        new.__class__ = Amr
        new.node_to_concepts = {} 
        new.__cached_triples = None
        for node in self.node_to_concepts:
            if node in node_map:
                new.node_to_concepts[node_map[node]] = self.node_to_concepts[node]
            else: 
                new.node_to_concepts[node] = self.node_to_concepts[node]
        return new
        

    def find_nt_edge(self, label, index):       
        for p,r,c in self.triples():
            if type(r) is NonterminalLabel:
                if r.label == label and r.index == index:
                    return p,r,c    
        for edge in self.nonterminal_edges():
            print edge
        print str(self)



    def remove_fragment(self, dag):
        """
        Remove a collection of hyperedges from the DAG.
        """
        res_dag = Amr.from_triples([edge for edge in self.triples() if not dag.has_edge(*edge)], dag.node_to_concepts)
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
        boundary_map = dict([(x, leaves[external[x]]) for x in external])
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
        res_dag.node_to_concepts.update(new_dag.node_to_concepts)
        return res_dag



if __name__ == "__main__":

    import doctest
    doctest.testmod()



      
