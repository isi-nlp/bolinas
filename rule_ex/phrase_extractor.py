from common.hgraph.hgraph import Hgraph, StrLiteral
from common.hgraph.hgraph import ListMap
from common.hgraph.amr_graph_description_parser import GraphDescriptionParser 
from common.hgraph.amr_corpus_reader import metadata_amr_corpus
from collections import defaultdict, deque

from common.cfg import NonterminalLabel
from common.rule import Rule
from parser.vo_rule import VoRule # Either rule class would do. 
from lib import fancy_tree
import sys
import itertools
from functools import reduce

import cStringIO

import operator    

import cPickle as pickle

from forest import Forest

MAX_CHART_SIZE = 5000 

    
class DecompositionForest(dict):  # This is a mapping from Partition objects to a list of other Partition objects. 
    pass
    
    def __init__(self):
        self.inconsistent_alignment = False


class IncompatibleAlignmentException(Exception):                    
    pass

class ChartTooBigException(Exception):                    
    pass


class Item(object):
    def __init__(self, rule):
        self.rule = rule
    def __hash__(self):
        return hash(self.rule)
    def __eq__(self, other): 
        return self.rule == self.other

        

class Rule(object): 
    def __init__(self, rule_id, symbol, rhs1, rhs2, count=0):
        self.rule_id = rule_id
        self.symbol = symbol
        self.logprob = 1.0
        self.count = count
        self.rhs1 = rhs1
        self.rhs2 = rhs2
        self.hashcache = None

    def _repr_svg_(self): 
        """
        IPython rich display method
        """
        graph = self.rhs1._get_gv_graph()
        graph.graph_attr['label']= "%s -> %s" % (self.symbol, " ".join([str(x) for x in self.rhs2]))
        graph.graph_attr['labelloc']= 't'
        output = cStringIO.StringIO()
        graph.draw(output, prog="dot", format="svg")
        return output.getvalue()
 

    def __str__(self): 
        return "%s -> %s | %s ; # Rule %i" % (self.symbol, " ".join([str(x) for x in self.rhs2 ]), self.rhs1.to_string(newline=False), self.rule_id)

    def __hash__(self):
        if self.hashcache: 
            return self.hashcache
        else: 
            self.hashcache = 7 * hash(self.symbol) + 43 * hash(self.rhs1) + 37 * hash(self.rhs2) 
            return self.hashcache
   
    def __eq__(self, o): 
        return self.symbol == o.symbol and self.rhs1 == o.rhs1 and self.rhs2 == o.rhs2


class Partition(object):
    
    def __init__(self, phrase, str_start, str_end, edges):
        self.phrase = phrase
        self.edges = edges
        self.str_start = str_start
        self.str_end = str_end       

    def __repr__(self):
        return "<%s,%i:%i,%i>"% (self.phrase, self.str_start, self.str_end, len([x for x in self.edges if x ==1]))

    def __hash__(self):
        return 19 * hash(self.edges) + 41 * self.str_start + 59 * self.str_end + 73 * hash(self.phrase)

    def __eq__(self,other): 
        return self.edges == other.edges and self.str_start == other.str_start and self.str_end == other.str_end and self.phrase == other.phrase

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_graph(self, graph):
        trips = graph.triples()
        return Hgraph.from_triples([trips[i] for i in range(len(trips)) if self.edges[i] == 1],{}, warn=False)

class PhraseExtractor(object):
    """
    Implementation of the SynSem algorithm described in the Coling 2012 paper.
    The output of this algorithm is a chart containing all possible grammar rules
    that are consistent with an alignment, such that each rule corresponds to a phrase
    in the syntactic representation.
    Different methods/heuristics can be used to extract a grammar from this chart.
    """

    def __init__(self):
        pass
        self.rules = {}
        self.rev_rules = {}
        self.rule_id_ctr = 0
            
    def add_rule(self, rule):
        try:
            rule_id = self.rules[rule]  
        except KeyError,e: 
            rule_id = self.rule_id_ctr
            rule.rule_id = rule_id 
            self.rules[rule] = rule_id 
            self.rule_id_ctr+=1
            self.rev_rules[rule_id] = rule
        return rule_id
        
    def save_grammar(self, f):
        for rid in range(len(self.rev_rules)):
            f.write(str(self.rev_rules[rid]))
            f.write("\n")
 
    def extract_from_corpus(self, hgraph_file, tree_file, chart_file, grammar_file):
        count = 1 
        
        for hgraphandmeta, pstree in self.parallel_amr_ptb_iterator(hgraph_file, tree_file):
            hgraph = hgraphandmeta.amr
            tokens = [t.rsplit("-",1)[0] for t in hgraphandmeta.tagged.split(" ")]
            tree = pstree[0].preprocess()
            
            graph_tokens_to_tree_tokens = normalize_tokenization(tree.leaves(), tokens)
            if not graph_tokens_to_tree_tokens: 
                print "Tokenization error", count
            
            new_alignments = {}
            for g in hgraph.node_alignments:
                alignments_for_g = []
                for tree_tok in hgraph.node_alignments[g]:
                    alignments_for_g.append(graph_tokens_to_tree_tokens[tree_tok])
                new_alignments[g] = alignments_for_g
            hgraph.node_alignments = new_alignments

            tree = tree.pos_tree()
            
            graph = hgraph.to_instance_edges()
            graph = replace_instance_edges(graph, tree)

            print count
            res = self.extract_from_pair(graph, tree)
            print "Done"
            chart = self.partition_chart_to_rule_chart(res)
            chart.sentence_id = hgraphandmeta.sent_id
            #chart_file.write("# ::id %s\n" % hgraphandmeta.sent_id)
            #chart_file.write(str(chart))
            #chart_file.write("\n\n")
            
            pickle.dump(chart, chart_file) 
            count += 1

        self.save_grammar(grammar_file)

    def parallel_amr_ptb_iterator(self, hgraph_file, tree_file):
        """
        Assume that there are possibly more trees than hypergraphs. Compare the strings and skip
        graphs that don't exist. There are as many alignments as hypergraphs.
        """
        
        hgraphs = metadata_amr_corpus(hgraph_file)
        trees = (fancy_tree.FancyTree(line).preprocess() for line in tree_file)    
         
        next_tree = trees.next()
        alignment = None
        for hgraph in hgraphs: 
            hgraph_str = "".join(x.rsplit("-",1)[0] for x in hgraph.tagged.split())
            while hgraph_str  != "".join(next_tree.leaves()): 
                next_tree = trees.next()
            yield hgraph, next_tree
            try:
                next_tree = trees.next()
            except StopIteration, e: 
                pass
 
    def partition_chart_to_rule_chart(self, chart): 

        graph_edge_list = chart.graph.triples()
        node_order = chart.graph.get_ordered_nodes()

        result = Forest()
        seen = {}  

        fragment_counter = [0] 

        def convert_chart(partition, external_nodes, nt, first = False):  
            nt = NonterminalLabel(nt.label) # Get rid of the index
  
            if partition in seen: 
                node = seen[partition]
                result.use_counts[node] += 1
                return node

            leaves = chart.tree.leaves()

            edges_in_partition = [graph_edge_list[i] for i in range(len(partition.edges)) if partition.edges[i] == 1]
            
            if not partition in chart:  # leaf

                graph = Hgraph.from_triples(edges_in_partition, {}, warn=False)
                graph.roots = graph.find_roots()
                graph.roots.sort(lambda x,y: node_order[x]-node_order[y])
                graph.external_nodes = external_nodes
                str_rhs = [leaves[i] for i in range(partition.str_start, partition.str_end+1)]
                rule = Rule(0, nt.label, graph, tuple(str_rhs), 1)
                rule_id =  self.add_rule(rule)
                fragment = fragment_counter[0]
                result[fragment] = [(rule_id,[])]
                result.use_counts[fragment] += 1
                seen[partition] = fragment
                fragment_counter[0] += 1
                return fragment


             
            poss = []   
            count = 0 
            for possibility in chart[partition]:
                count += 1
                partition_graph = Hgraph.from_triples(edges_in_partition,{}, warn=False) # This is the parent graph
                partition_graph.roots = partition_graph.find_roots()
                partition_graph.roots.sort(lambda x,y: node_order[x]-node_order[y])
                partition_graph.external_nodes = external_nodes
                children = []
                #print partition_graph.to_amr_string()



                spans_to_nt = {}
                old_pgraph = partition_graph

                index = 1
                for subpartition in possibility: #These are the different sub-constituents
                        
                    edges_in_subpartition = [graph_edge_list[i] for i in range(len(subpartition.edges)) if subpartition.edges[i] == 1]
                    if edges_in_subpartition: # Some constituents do not have any edges aligned to them
                        sub_graph = Hgraph.from_triples(edges_in_subpartition,{}, warn=False)
                        sub_graph.roots = sub_graph.find_roots()
                        sub_graph.roots.sort(lambda x,y: node_order[x] - node_order[y])
                        external_node_list = partition_graph.find_external_nodes2(sub_graph)
                        external_node_list.sort(lambda x,y: node_order[x] - node_order[y])
                        sub_external_nodes = dict([(k,v) for v,k in enumerate(external_node_list)])
                        sub_graph.external_nodes = sub_external_nodes
                        sub_nt = NonterminalLabel("%s%i" % (subpartition.phrase, len(sub_external_nodes)), index)
                        children.append(convert_chart(subpartition, sub_external_nodes, sub_nt)) # Recursive call
                        old_pgraph = partition_graph
                        partition_graph = partition_graph.collapse_fragment2(sub_graph, sub_nt, external = external_node_list, warn = False)
   
                        spans_to_nt[subpartition.str_start] = (sub_nt, subpartition.str_end)
                    else: 
                        sub_nt = NonterminalLabel(subpartition.phrase, index)
                           
                    #assert partition_graph.is_connected()
                    index += 1

                partition_graph.roots = partition_graph.find_roots()
                partition_graph.roots.sort(lambda x,y: node_order[x]-node_order[y])


                # Assemble String rule
                str_rhs = []
                i = partition.str_start
                while i<=partition.str_end: 
                    if i in spans_to_nt:
                        new_nt, i = spans_to_nt[i]
                        str_rhs.append(new_nt)
                    else:
                        str_rhs.append(leaves[i])        
                    i = i + 1

                rule = Rule(0, nt.label, partition_graph, tuple(str_rhs), 1)
                rule_id = self.add_rule(rule)
                        
                poss.append((rule_id, children))
              
            fragment = fragment_counter[0]
            result[fragment] = poss
            result.use_counts[fragment] += 1
            seen[partition] = fragment
            fragment_counter[0] += 1
            return fragment 
           
        result.root = convert_chart(chart.root, {}, NonterminalLabel(chart.root.phrase), first = True)             
        return result

    def extract_from_pair(self, hgraph, pstree, alignments = None):
        """
        Construct a chart from the given hypergraph, phrase structure tree,
        and alignment dictionary. If the alignment dictionary is not 
        specified, we assume the alignments are part of the hypergraph, which 
        should contain the instance variables edge_alignments and node_alignments.
        TODO: This should also work with n-best syntactic parses. There is little
        gold syntax for AMR bank and other resources.
        """
 
        if not alignments: 
            alignments = hgraph.edge_alignments

        #Alignments come in the form of a dictionary mapping edges to tuples of tokens. We also create the reverse mapping
        rev_alignments = defaultdict(list)
        for edge, tokens in alignments.items():
            for t in tokens:
                rev_alignments[t].append(edge)

        original_string = pstree.leaves() 

        chart = DecompositionForest()

        #graph_edges_to_index = dict([(k,v) for v,k in enumerate(graph.triples())])
        graph_edge_list = hgraph.triples()

        count = [0] 

        def compute_chart(tree,graph, prefix = ""): # Recursively compute the chart. Graph is the sub-graph we're considering, tree is the tree of spans.  
            count[0] += 1

            triples = set(graph.triples())
            edge_vector= tuple([1 if x in triples else 0 for x in graph_edge_list])

            leaves = tree.leaves()
            #if not isinstance(tree,fancy_tree.FancyTree):
            #    triples = set(graph.triples())
            #    edge_vector= tuple(1 for x in graph_edge_list if x in triples else 0)
            #    return Partition(tree.node, leaves[0], leaves[-1], edge_vector) 
            #else: 
            if len(tree) == 1 and not isinstance(tree[0],fancy_tree.FancyTree): 
                return Partition(tree.node, leaves[0], leaves[-1], edge_vector) 

            # First get the set of aligned edges for this constituent and it's children
            aligned_edges_for_span = set([edge for token in tree.leaves() for edge in rev_alignments[token]])

            partition_object = Partition(tree.node, leaves[0], leaves[-1], edge_vector) 
            if partition_object not in chart: 

                try:
                    possibilities = []
                    child_edgesets = []
                    # Compute edge set for each child
                    for t in tree: 
                        edgeset = []
                        for l in t.leaves():
                            edgeset.extend(rev_alignments[l])
                        child_edgesets.append(edgeset)

                    # For each possible partitioning  
                    for cparts in get_binarized_partitions(graph, child_edgesets):
                        child_forests = []
                        for i in range(len(tree)):
                            childgraph = Hgraph.from_triples(cparts[i],{}, warn= False)
                            sub_forest = compute_chart(tree[i], childgraph, prefix = prefix + " ")
                            if len(chart) > MAX_CHART_SIZE: 
                                raise ChartTooBigException, "Chart size exceeded 5000 entries. dropping this sentence."
                            child_forests.append(sub_forest)
                        possibilities.append(child_forests)
 
                    chart[partition_object] = possibilities
                except IncompatibleAlignmentException:
                    chart.inconsistent_alignment = (tree.node, leaves[0], leaves[-1])
                    
                    return partition_object
            return partition_object
            

        # Get a tree in which the leaves are replaced by token indices
        span_tree = pstree.span_tree() 
        chart.root = compute_chart(span_tree, hgraph)
        chart.graph = hgraph
        chart.tree= pstree
        if chart.inconsistent_alignment:
            sys.stderr.write("Warning: Incompatible alignments for phrase %s [%s,%s].\n" % chart.inconsistent_alignment)
        return chart
            

def get_binarized_partitions(graph, edgesets): 
    if len(edgesets) == 1:
        yield [graph.triples()]
        return
        
    gen = get_partitions(graph, edgesets[0], [edge for edgeset in edgesets[1:] for edge in edgeset])
    
    for left_edges, right_edges in gen: 

        possibilities = get_binarized_partitions(Hgraph.from_triples(right_edges,{},warn=False), edgesets[1:])
        poss_list = list(possibilities)
        for partitions in poss_list: 
            yield [left_edges] + partitions 
        

def get_line_graph(graph):
   
    lgraph = Hgraph()
    edges_for_node = defaultdict(list) 

    for p,r,ch in graph.triples():
        edges_for_node[p].append(str((p,r,ch)))
        for c in ch: 
            edges_for_node[c].append(str((p,r,ch)))

    for r in edges_for_node: 
        for p,c in itertools.combinations(edges_for_node[r],2):
            lgraph._add_triple(p,r,(c,),warn=False)
            lgraph._add_triple(c,r,(p,),warn = False)
    lgraph.roots = lgraph.find_roots()
    return lgraph


def get_initial_partitions(graph, p1_vertices, p2_vertices):

    if not p2_vertices: 
        return graph.keys(), []

    if not p1_vertices: 
        return [], graph.keys()

    part1, part2 = None, None

    res1 = get_spanning_tree(graph, p1_vertices, p2_vertices)
    if not res1:  # No partition possible
        raise IncompatibleAlignmentException, "No spanning tree possible for partition1." 
    
    res2 = get_spanning_tree(graph, p2_vertices, res1)
    if not res2: 
        res3 = get_spanning_tree(graph, p2_vertices, p1_vertices)
        if not res3: # No partition possible
            raise IncompatibleAlignmentException, "No spanning tree possible for partition2." 
            return False
        res4 = get_spanning_tree(graph, p1_vertices, res3)
        if not res4: # No partition possible?? 
            raise IncompatibleAlignmentException, "No spanning tree possible for partition1 after removing partition2." 
        else: 
            part1, part2 = res4, res3
    else: 
        part1, part2 = res1, res2


    part1_left = delete_nodes(graph, part2)
    part1.update(reachable_nodes(part1_left, part1))

    part2_left = delete_nodes(graph, part1)
    part2.update(reachable_nodes(part2_left, part2))
    return part1, part2 
    

def reachable_nodes(graph, nodes):
    if not nodes: 
        return set()
    # Run DFS
    s = get_any(nodes)
    stack = [s] 
    tabu = set()
    tabu.add(s)
    while stack:
        n = stack.pop()
        for (p,r,ch) in graph.out_edges(n):
            c = ch[0]
            if not c in tabu: 
                stack.append(c)
                tabu.add(c)
    return tabu
 

def get_any(s):
    for x in s: 
        return x
   
def get_spanning_tree(graph, nodes, initial_tabu = set()):
    # Run DFS on until all vertices in nodes have been found, avoiding vertices in tabu 
    tabu = set(initial_tabu)

    if not nodes: 
        nodes = set(set(graph.keys()) - tabu)
    s = get_any(nodes) 
    parent = {s: None} # Keep track of the tree by recording the parent for each node
    # Run DFS
    stack = [s]
    tabu.add(s)
    while stack:
        n = stack.pop()
        for (p,r,ch) in graph.out_edges(n):
            c = ch[0]
            if not c in tabu: 
                parent[c] = p
                stack.append(c)
                tabu.add(c)

    if not set(nodes) <= tabu:  #No partitioning possible because removing p2 alignments disconnected p1 alignments
        return set() 

    # Extract a spanning tree containing only the vertices in p1
    res_set = set()
    n = None
    for x in nodes: 
        n = x
        while (n!=s):
            res_set.add(n)
            if not n in parent:
                return set() 
            n = parent[n]
    res_set.add(s) 
    return res_set

def delete_nodes(graph, nodes):
    g = Hgraph()
    for p,r,ch in graph.triples(): 
        if (p not in nodes) and (not (len(ch)==1 and ch[0] in nodes)):
            g._add_triple(p,r,ch, warn= False)
        else:
            if (p not in nodes) and (p not in g):
                g[p] =  ListMap() 
            if len(ch) == 1 and (ch[0] not in nodes) and (ch[0] not in g):
                g[ch[0]] =  ListMap()
    g.roots = g.find_roots(warn = False)
    return g


def get_partitions(graph, p1edges, p2edges):
    """
    A generator that lists all possible partitions of the graph consistent with the initial edges. 
    Initial edges is a list of lists of edges known to be in each partition.
    """
    g2 = get_line_graph(graph)
     
    p1vertices = [str(x) for x in p1edges]
    p2vertices = [str(x) for x in p2edges]

    p1,p2 = get_initial_partitions(g2, p1vertices, p2vertices) 

    pvector = []
    vertexidx = {}
    rev_vertexidx = {}
    for i,v in enumerate(g2.keys()):
        pvector.append(0)
        vertexidx[v] = i   
        rev_vertexidx[i] = v

    for v in p1: 
        pvector[vertexidx[v]] = 1 
    for v in p2: 
        pvector[vertexidx[v]] = 2 
    

    pvector = tuple(pvector) 
    seen = set()
    queue = [pvector]

    while queue: 
        pvector = queue.pop(0)
        if pvector in seen: 
            continue 
        seen.add(pvector)
        partition1 = set([rev_vertexidx[i] for i in range(len(pvector)) if pvector[i]==1])
        partition2 = set([rev_vertexidx[i] for i in range(len(pvector)) if pvector[i]==2])
        
        yield [eval(x) for x in partition1], [eval(y) for y in partition2]

        gp1 = delete_nodes(g2,partition2)
        gp2 = delete_nodes(g2,partition1)
        #if not gp1 or not gp2:
        #    continue
        safe = set(p1vertices) | set(p2vertices) # Nodes we cannot move: Everything that's aligned and everything that's an articulation point.
        safep1 = compute_articulation_points(gp1)
        safep2 = compute_articulation_points(gp2)
        safe.update(safep1)
        safe.update(safep2)

        # Get possible nodes that can be moved
        possible_to_p2 = set()
        possible_to_p1 = set()
        for p,r,ch in g2.triples(): #all nodes that aren't safe and that are boundary nodes 
            #print p, ch[0], p in safe, ch[0] in safe
            c = ch[0]
            if not p in safe: 
                if (p in partition1 and c in partition2):
                    possible_to_p2.add(p) 
                elif (p in partition2 and c in partition1):
                    possible_to_p2.add(p) 

                #if not c in safe:
                #    print "trying ",p,c  
                #    if (p in p1 and c in p2):
                #        possible_to_p2.append(p) 
                #    elif (p in p2 and c in p1): 
                #        possible_to_p1.append(p) 
        
        #print "Possible to p2", possible_to_p2
        #print "Possible to p1", possible_to_p1

        for n in possible_to_p1: 
            newpvector = list(pvector)
            newpvector[vertexidx[n]] = 1
            newpvector = tuple(newpvector)
            if not newpvector in seen: 
                queue.append(newpvector)

        for n in possible_to_p2: 
            newpvector = list(pvector)
            newpvector[vertexidx[n]] = 2
            newpvector = tuple(newpvector)
            if not newpvector in seen: 
                queue.append(newpvector)
          

def count_possibilities(forest):
    count = 0 
    for p in forest.poss: 
       if len(p)>0 and isinstance(p[0], Possibilities):
           count += reduce(operator.mul, [count_possibilities(c) for c in p],1)
       else:
           return 1
    return count
            

def compute_articulation_points(graph): 
    # Hopcroft & Tarjan's algorithm 
    parent = {}
    depth = {}
    low = {}

    articulation_points = set()

    if len(graph.keys()) == 0:
        return set()

    s = get_any(graph.keys())
    tabu = set()
   
    def compute_articulation_points_rec(n, d):
        tabu.add(n)
        depth[n] = d
        low[n] = d
        child_count = 0
        is_articulation = False
        for (p,r,ch) in graph.out_edges(n): 
            if ch: 
                c = ch[0]
                if not c in  tabu:
                    parent[c] = n
                    compute_articulation_points_rec(c, d+1)
                    child_count = child_count + 1
                    if low[c] >= depth[n]:
                        is_articulation = True
                    low[n] = min(low[n], low[c])
                else: 
                    if not n in parent or c != parent[n]:
                        low[n] = min(low[n], depth[c])
                        
        if (n in parent and is_articulation) or (n==s and child_count > 1):
            articulation_points.add(n)
        
    compute_articulation_points_rec(s,0)
    return articulation_points



def render_partition(g, p1, p2): 
        """
        IPython rich display method
        """
        graph = gv_partition_graph(g,p1 ,p2)
        output = cStringIO.StringIO()
        graph.draw(output, prog="dot", format="svg") 
        return output.getvalue()

def gv_partition_graph(self, p1, p2, instances = True):
        """
        Return a pygraphviz AGraph.
        """
        import pygraphviz as pgv
        
        graph = pgv.AGraph(strict=False,directed=True)
        graph.node_attr.update(height=0.1, width=0.1, shape='none', fontsize='9')
        graph.edge_attr.update(fontsize='9')
        counter = 0
        counter2 = 0
        for edge in self.triples(instances):
            node, rel, child  = edge
            if edge in p1: 
                edgecolor = "red"
            elif edge in p2: 
                edgecolor = "black"
            else: 
                edgecolor = "white"
            if node in self.node_to_concepts and self.node_to_concepts[node]:
                graph.add_node(node, label=self.node_to_concepts[node], style="filled", fillcolor = ("black" if node in self.external_nodes else "white"), fontcolor = ("white" if node in self.external_nodes else "black"))
            if isinstance(node,StrLiteral):
                node = str(node)
                graph.add_node(node, label='"%s"' % node)
            if len(child) > 1:
                centernode = "hedge%i" % counter
                counter += 1
                graph.add_node(centernode, shape="point", label="", width="0", height="0")
                graph.add_edge(node, centernode, dir="none", label="%s"%rel, color=edgecolor)
                for tail in  child: 
                    if tail in self.node_to_concepts and self.node_to_concepts[tail]:
                        graph.add_node(tail, label=self.node_to_concepts[tail], style="filled", fillcolor = ("black" if tail in self.external_nodes else "white"), fontcolor = ("white" if node in self.external_nodes else "black"))
                    if isinstance(node, StrLiteral): 
                        tail = str(tail)
                        graph.add_node(tail, label='"%s"' % tail)
            
                    graph.add_edge(centernode, tail, color=edgecolor)
            else: 
                if child: 
                    nodestr, tail = node, child[0]
                else: 
                    graph.add_node("#@"+str(counter2),label="")
                    nodestr, tail = node,  "#@"+str(counter2)
                    counter2 +=1
    
                if tail in self.node_to_concepts and self.node_to_concepts[tail]:
                    graph.add_node(tail, label=self.node_to_concepts[tail], style="filled", fillcolor = ("black" if tail in self.external_nodes else "white"), fontcolor = ("white" if tail in self.external_nodes else "black"))
                if isinstance(tail, StrLiteral): 
                    tail = str(tail)
                    graph.add_node(tail, label='"%s"' % tail)
                graph.add_edge(nodestr, tail, label="%s"%rel, color=edgecolor)
        return graph





def test():
    tree = FancyTree("""
    (S
        (NP (DT The) (NN boy))
        (VP (VBZ wants)
          (NP (DT the) (NN girl)
            (S
              (VP (TO to)
                (VP (VB believe)
                  (NP (PRP him)))))))
        (. .))""")
    graph = Hgraph.from_string("(w.want :arg0 b.boy :arg1 (b2.believe :arg0 (g.girl) :arg1 b.))")
    graph.node_alignments = {"b":[1], "w":[2], "g":[4], "b2":[6]}
    graph = graph.to_instance_edges()




def replace_instance_edges(graph, tree):
    t = []
    alignments = {}
    tree_leaves = tree.leaves()
    for e in graph.triples():
        
        if e in graph.edge_alignments:
            p,r,ch = e 
            token = graph.edge_alignments[e][0] # TODO: not sure what to do with multiple tokens
            new_edge = (p,"%s'" % tree_leaves[token],ch)
            t.append(new_edge)
            alignments[new_edge] = [token]
        else: 
            t.append(e)
    res = Hgraph.from_triples(t,{},warn=False)
    res.edge_alignments = alignments 
    res.node_alignments = graph.node_alignments
    res.roots = graph.roots
    res.external_nodes = graph.external_nodes
    return res


def normalize_tokenization(s1, s2): 
    t2 = 0
    tok2 = ""
    s1tos2map = defaultdict(list)
    s2tos1map = {}
    for t1 in range(len(s1)):
        tok1 = s1[t1].lower()
        tok2 = ""
        while not tok2==tok1: 
            tok2 += s2[t2].lower()
            if len(tok2) > len(tok1):
                return None
            s1tos2map[t1].append(t2) 
            s2tos1map[t2] = t1
            t2 += 1
    return s2tos1map
        
    
def main():
    ex = PhraseExtractor()
    #return ex, ex.extract_from_corpus(open(sys.argv[1],'r'), open(sys.argv[2],'r'), open(sys.argv[3],'w'), open(sys.argv[4],'w'))
    return ex.parallel_amr_ptb_iterator(open(sys.argv[1],'r'), open(sys.argv[2],'r')), ex
    

if __name__ == "__main__": 


    #ex, chart = main()
    #sys.exit(0)   
 
    #it, ex = main()

    #amr, tree = it.next()
    #graph = amr.amr
    #tree = tree[0].preprocess()
    #tree = tree.pos_tree()
    #graph = graph.to_instance_edges()
    #graph = replace_instance_edges(graph, tree)
    #res = ex.extract_from_pair(graph, tree)
    #chart1 = ex.partition_chart_to_rule_chart(res)
    #print chart.get_rule_counts()
    ex = PhraseExtractor()

    p = GraphDescriptionParser()
    graph = p.parse_string("""(c~0 / care-01 :ARG0 y :ARG1 (e~2 / explain-01 :ARG0 (y~4 / you) :ARG1 (t2~5 / thing :ARG0-of (c2~3 / cause-01 :ARG1 (t / think-01 :ARG0 y :ARG1 (i~9 / idiot :domain (h~6 / he)))))) :mode (interrogative))""")
    tree = fancy_tree.FancyTree("(SQ (VP (VB Care) (VP (TO to) (VP (VB explain) (SBAR (WHADVP (WRB why)) (S (NP (PRP you)) (VP (VBP think) (S (NP (PRP he)) (VP (VBZ is) (NP (DT an) (NN idiot)))))))))) (-PERIOD- ?))")

    #amr, tree = it.next()
    #amr, tree = it.next()
    #amr, tree = it.next()
    #graph = amr.amr
    #tree = tree[0].preprocess()
    #tree = tree.pos_tree()
    graph = graph.to_instance_edges()
    #graph = replace_instance_edges(graph, tree)
    res = ex.extract_from_pair(graph, tree)
    #chart2 = ex.partition_chart_to_rule_chart(res)
    #print chart2.get_rule_counts()

    #from lib.fancy_tree import FancyTree
    #tree = FancyTree("""
    #(S
    #    (NP (DT The) (NN boy))
    #    (VP (VBZ wants)
    #      (NP (DT the) (NN girl)
    #        (S
    #          (VP (TO to)
    #            (VP (VB believe)
    #              (NP (PRP him)))))))
    #    (. .))""")
    ##print tree
    #graph = Hgraph.from_string("(w.want :arg0 b.boy :arg1 (b2.believe :arg0 (g.girl) :arg1 b.))")
    #graph.node_alignments = {"b":[1], "w":[2], "g":[4], "b2":[6]}

    #graph = graph.to_instance_edges()
    #print tree
    #print graph
    ###print graph.to_string(newline = False)
    #
    #ex.save_grammar(open(sys.argv[3],'w'))
    #print count_possibilities(res)        
    #g = g.to_concept_edge_labels()
    #del g.edge_alignments[('g', 'ARG0-of:publish-01', ('p2',))]
    #res =  ex.extract_from_pair(g,t) 
    #ex.get_rules_from_tree2(g,t.leaves(),res)
    #g = Hgraph.from_string("(a. :ab (b. :bd d. :bc (c. :cf g.) :bf f.) :ae (e. :ef f.) :ad (d. :df (f. :fg (g. :gh h.))))")

    #chart = ex.extract_from_pair(graph, tree)
   
    ##res, g2 = get_all_simple_paths(g, [('a','ad',('d',)), ('g','gh',('h',)), (('b', 'bc', ('c',)))]) 
    #g2
    #for p in res: 
    #    print p

