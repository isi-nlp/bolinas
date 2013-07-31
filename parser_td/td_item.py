from common.hgraph.hgraph import NonterminalLabel
from common import log
import sys
from collections import defaultdict as ddict

class Subgraph(object):
    pass

class BoundarySubgraph(Subgraph):
  """
  Represents a boundary-induced subgraph, for fast union computation.
  """
  def __init__(self, boundary_nodes, boundary_edges, size):
    self.boundary_nodes = frozenset(boundary_nodes)
    self.boundary_edges = ddict(frozenset, boundary_edges) # maps boundary nodes to their edges
    self.size = size

    self.hedges = frozenset([(k,boundary_edges[k]) for k in boundary_edges])
    self.saved_hash = 17 * hash(self.boundary_nodes) + 23 * hash(self.hedges)

  def __hash__(self):
    return self.saved_hash

  def __eq__(self, other):
    return isinstance(other, Subgraph) and \
        self.boundary_nodes == other.boundary_nodes and \
        self.hedges == other.hedges
        # Need to check boundary edges too?

  def __repr__(self):
    return "[%s, %d]" %(self.boundary_nodes, self.size)

  def is_member(self, node, graph):
    """
    Tests if node is a member of this subgraph.
    """
    if node in self.boundary_nodes:
      return True
    min_length = sys.maxint
    min_edge = None
    for boundary_node in self.boundary_nodes:
      if (node, boundary_node) not in graph.fw_table:
        continue
      length, edge = graph.fw_table[(node, boundary_node)]
      if length < min_length:
        min_length = length
        min_edge = edge
        min_boundary_node = boundary_node

    if not min_edge:
      return False
    return min_edge in self.boundary_edges[min_boundary_node]

  def __len__(self):
    return self.size


class ExplicitSubgraph(Subgraph): 
  """
  A subgraph that is represented by an explicit list of nodes and hyperedges.
  """
  def __init__(self, nodes, edges):
    self.nodes = frozenset(nodes)
    self.edges = frozenset(edges)

    self.saved_hash = 17 * hash(self.nodes) + 23 * hash(self.edges)

  def __hash__(self):
    return self.saved_hash

  def __eq__(self, other):
    return isinstance(other, ExplicitSubgraph) and \
        self.nodes == other.nodes and \
        self.edges == other.edges

  def __repr__(self):
    return "Subgraph[%s, %d]" %(self.nodes, len(self.edges))

  def __len__(self):
    return len(self.edges) 


class Item(object):
  """
  Represents a single chart item, which is fully specified by an induced
  subgraph, a rule, a node of that rule's tree decomposition, and a bijection
  between nodes of the rule and the input graph.
  """

  TERMINAL = 0
  NONTERMINAL = 1
  BINARY = 2
  ROOT = 3
  NONE = 4

  def __init__(self, rule, tree_node, graph, subgraph=None, mapping=None, nodelabels = False):
    self.rule = rule
    self.tree_node = tree_node
    self.graph = graph
    self.nodelabels = nodelabels

    # tree_node specifies the subtree of the tree decomposition which has
    # *already* been recognized

    # Every node of the tree decomposition will be one of
    # 1. A leaf
    # 2. A node introducing a single edge (either terminal or nonterminal)
    # 3. A binary node
    # The action taken by the parser on an item is determined by that item's
    # parent in the tree. If there is no parent, we've recognized the whole rule
    # and (implicitly) invoke the Root rule to become a passive item. If the
    # parent introduces an edge, we want to invoke the Terminal or Nonterminal
    # rule as appropriate. If the parent is a binary node, we want the first
    # child to look up its sibling and invoke the Binary rule, and the second
    # child to trigger no action at all.
    # This logic is encoded in the block that follows.
    
    # target specifies the next action the parser should perform on this item
    # self_key is the key by which this item will be indexed
    # next_key indexes the item we want to combine with this one
    #   (my next_key is that item's self_key)
    if tree_node in self.rule.tree_second_siblings:
      self.target = Item.NONE
      self.self_key = (self.rule.rule_id, self.tree_node)
    elif tree_node not in self.rule.tree_to_parent:
      self.target = Item.ROOT
      self.closed = True
      self.self_key = self.rule.symbol
    elif tree_node in self.rule.tree_to_sibling:
      self.target = Item.BINARY
      self.next_key = (self.rule.rule_id, self.rule.tree_to_sibling[self.tree_node])
    else:
      search_edge = \
          self.rule.tree_to_edge[self.rule.tree_to_parent[self.tree_node]]
      if isinstance(search_edge[1], NonterminalLabel):
        self.target = Item.NONTERMINAL
        self.next_key = search_edge[1].label
        self.outside_symbol = search_edge[1].label
        self.outside_index = search_edge[1].index
      else:
        self.target = Item.TERMINAL
        self.next_key = search_edge[1]
      self.next_key_edge = search_edge

    # initialize empty if we haven't been given a subgraph and mapping
    self.subgraph = subgraph
    if not self.subgraph:
      self.subgraph = self.init_subgraph()
    self.mapping = mapping
    if not self.mapping:
      self.mapping = {}
    self.rev_mapping = dict((val,key) for key, val in self.mapping.items())

  def __eq__(self, other):
    return isinstance(other, Item) and \
        other.rule == self.rule and \
        other.tree_node == self.tree_node and \
        other.subgraph == self.subgraph and \
        other.mapping == self.mapping 

  def __hash__(self):
    return 13 * self.rule.rule_id + \
           17 * self.tree_node + \
           23 * hash(self.subgraph)  

  def __repr__(self):
    return '[%d, %s, %d, %s, {%d/%d}, (%d)]' % (self.rule.rule_id, self.rule.symbol, self.tree_node,\
        self.next_key_edge if "next_key_edge" in self.__dict__ else None, len(self.subgraph), len(self.rule.rhs1.triples()), self.target)

  ## BEGIN EXPLICIT REPR

  # these methods can be overriden to use a more compact subgraph representation
  # (e.g. boundary nodes)

  def init_subgraph(self):
    """
    Produces the object representing an empty subgraph.
    """
    return ExplicitSubgraph(set(), set())

  def matches_whole_graph(self):
    """
    Determines whether this item has recognized the entire input graph.
    """
    return len(self.subgraph) == len(self.graph.triples())

  def check_subgraph_overlap(self, oitem):
    """
    Determines whether this item and oitem recognize disjoint subgraphs.
    If so, returns the union of both subgraphs.
    """

    # Originally this method just checked for edge overlap. This is insufficient
    # as nodes with two adjacent edges (that are not external nodes) can be 
    # touched twice by different rules. We need to make sure that a rule
    # can only apply if the shared graph node maps to an external node of the rule.

    # Get nodes corresponding to external nodes of the other item.
    real_other_ext = [oitem.mapping[x] for x in oitem.rule.boundary_nodes]
    real_other_ext.append(oitem.mapping[oitem.rule.root_node])
    if any((self.subgraph.nodes & oitem.subgraph.nodes) - set(real_other_ext)):
        return None

    if any(self.subgraph.edges & oitem.subgraph.edges):
        return None
    
    nsubgraph = ExplicitSubgraph(self.subgraph.nodes | oitem.subgraph.nodes, self.subgraph.edges | oitem.subgraph.edges)
    return nsubgraph

  def check_edge_overlap(self, edge):
    """
    Determines whether edge overlaps with this item's recognized subgraph.
    If not, return the union of the subgraph and edge.
    """
    if self.nodelabels:
        head = edge[0][0]
        nodes = [head]
        for n in edge[2]:
            nodes.append(n[0])
        new_edge = (head, edge[1], tuple(nodes))
    else:
        new_edge = edge
        if new_edge in self.subgraph.edges:
          return None
        nodes = [edge[0]]
        nodes.extend(edge[2])

    if new_edge in self.subgraph.edges:
        return None

    nsubgraph = ExplicitSubgraph(self.subgraph.nodes | set(nodes), self.subgraph.edges | set([new_edge]))
    return nsubgraph

  ## END

  def check_mapping_bijection_nonterminal(self, oitem):
    """
    Determines whether this item's node mapping and oitem's, when merged, still
    form a valid bijection. (oitem is passive).  If so, returns the resulting
    map.
    """
    # create a copy of my mapping
    nmapping = dict(self.mapping)

    # oitem is a passive item with a different rule, so the node names in its
    # mapping are totally irrelevant
    # we're matching it against one edge in this rule, so we need to make sure
    # that the head and tails line up properly.

    # pull out the roots of oitem, and the piece of my rule that it corresponds
    # to
    # myhead is NOT the root of my rule! it's the head node of the edge I'm
    # matching
    if self.nodelabels:
        myhead = self.next_key_edge[0][0]
    else: 
        myhead = self.next_key_edge[0]

    # if oitem has mapped its root, and I have mapped the corresponding head
    # node in my rule, make sure the two mappings agree
    oroot = oitem.rule.root_node
    graph_root = oitem.mapping[oroot]
    if myhead in self.mapping and self.mapping[myhead] != graph_root:
      return None
    nmapping[myhead] = graph_root

    # now repeat that procedure for every tail node
    for i in range(len(self.next_key_edge[2])):
      if self.nodelabels:
          mynode = self.next_key_edge[2][i][0]
      else: 
          mynode = self.next_key_edge[2][i]
      onode = oitem.rule.rhs1.rev_external_nodes[i]
      graph_node = oitem.mapping[onode]
      if mynode in self.mapping and self.mapping[mynode] != graph_node:
          return None
      nmapping[mynode] = graph_node
 
    if len(nmapping.keys()) != len(set(nmapping.values())):
        return None
 
    return nmapping

  def check_mapping_bijection_binary(self, oitem):
    """
    As in check_mapping_bijection_nonterminal, but for active oitem.
    """
    # Unlike above, oitem has the same rule (and so the same node names) that I
    # do
    # I just make sure the mappings agree in both directions
    for node in self.mapping:
      if node in oitem.mapping:
        if self.mapping[node] != oitem.mapping[node]:
          return None
    nmapping = dict(self.mapping)
    for onode in oitem.mapping:
      if onode in self.mapping:
        if oitem.mapping[onode] != self.mapping[onode]:
          return None
      nmapping[onode] = oitem.mapping[onode]
    
    # Also need to make sure this is really a bijection
    if len(nmapping.keys()) != len(set(nmapping.values())):
        return None

    return nmapping

  def check_mapping_bijection_terminal(self, edge):
    """
    As in the previous method, but with a terminal edge rather than a chart
    item.
    """
    # this time there's only one edge to worry about, so I just need to check
    # for internal consistency

    if self.nodelabels:
        head = self.next_key_edge[0][0]
        edgehead = edge[0][0]
        if self.next_key_edge[2]:
            tails, labels = zip(*self.next_key_edge[2])
        else: 
            tails, labels = [], []
        if edge[2]:
            edgetails, edgelabels = zip(*edge[2])
        else:
            edgetails, edgelabels = [], [] 
    else:
        head = self.next_key_edge[0]
        edgehead = edge[0]
        tails = self.next_key_edge[2]
        edgetails = edge[2]
    
    if head in self.mapping:
      if self.mapping[head] != edgehead:
        return None
    nmapping = dict(self.mapping)
    nmapping[head] = edgehead

    for i in range(len(tails)):
      if tails[i] in self.mapping:
        if self.mapping[tails[i]] != edgetails[i]:
          return None
      nmapping[tails[i]] = edgetails[i]
    if len(nmapping.keys()) != len(set(nmapping.values())):
        return None
    return nmapping

  def terminal(self, edge):
    """
    Attempts to apply the (unary) Terminal rule and consume edge. Returns the resulting
    item, if the attempt succeeded.
    """
    
    # This is essentially the "shift" operation
    if self.target != Item.TERMINAL:
      return None
    if edge[1] != self.next_key:
      return None
    if len(edge[2]) != len(self.next_key_edge[2]):
      return None

    if self.nodelabels: #make sure labels agree 
        nhead, nheadlabel = self.next_key_edge[0]
        oh, ohlabel = edge[0]
        if nheadlabel != ohlabel:
            return None
        if self.next_key_edge[2]:
            ntail, ntaillabels = zip(*self.next_key_edge[2])
        else: ntail, ntaillabels = [],[]
        if edge[2]:
            otail, otaillabels = zip(*edge[2])
        else: 
            otail, otaillabels = [], []
        if ntaillabels != otaillabels: 
            return None

    nsubgraph = self.check_edge_overlap(edge)
    if not nsubgraph:
      return None

    nmapping = self.check_mapping_bijection_terminal(edge)
    if not nmapping:
      return None

    # if I get subclassed, I have to make sure I create new instances of the
    # subclass
    return self.__class__(self.rule,
        self.rule.tree_to_parent[self.tree_node],
        self.graph,
        nsubgraph,
        nmapping, nodelabels = self.nodelabels)

  def nonterminal(self, oitem):
    """
    Attempts to apply the (unary) Nonterminal rule and consume oitem, returning the
    result if successful.
    """

    if self.target != Item.NONTERMINAL:
      raise TypeError, "%s is not a nonterminal." % str(self)
    if oitem.target != Item.ROOT:
      raise TypeError, "%s is not at the root of its tree decomposition." %str(oitem)
    if oitem.rule.symbol != self.next_key:
      log.debug('symbol mismatch')
      return None

    if len(oitem.rule.rhs1.external_nodes) != len(self.next_key_edge[2]):
        log.debug('hyperedge type mismatch')
        return None

    nsubgraph = self.check_subgraph_overlap(oitem)
    if not nsubgraph:
      log.debug('overlap')
      return None

    nmapping = self.check_mapping_bijection_nonterminal(oitem)
    if not nmapping:
      log.debug('bijection')
      return None

    return self.__class__(self.rule,
        self.rule.tree_to_parent[self.tree_node],
        self.graph,
        nsubgraph,
        nmapping, nodelabels = self.nodelabels)

  def binary(self, oitem):
    """
    Attempts to apply the Binary rule and consume oitem, returning the result if
    successful.
    """
    if self.target != Item.BINARY:
      return None
    if oitem.target != Item.NONE:
      return None
    if self.rule.tree_to_sibling[self.tree_node] != oitem.tree_node: 
      return None

    nsubgraph = self.check_subgraph_overlap(oitem)
    if not nsubgraph:
      return None
    nmapping = self.check_mapping_bijection_binary(oitem)
    if not nmapping:
      return None

    return self.__class__(self.rule,
        self.rule.tree_to_parent[self.tree_node],
        self.graph,
        nsubgraph,
        nmapping, nodelabels = self.nodelabels)


class BoundaryItem(Item):
  """
  A drop-in replacement for the Item class which uses a boundary representation
  rather than a list of all the recognized edges.
  """
  def is_disjoint(self, osubgraph):
    """
    Checks whether my subgraph and osubgraph are disjoint.
    """
    for node in self.subgraph.boundary_nodes:
      if node in osubgraph.boundary_nodes:
        if len(self.subgraph.boundary_edges[node] & \
            osubgraph.boundary_edges[node]) != 0:
          return False        
      elif osubgraph.is_member(node, self.graph): 
        return False
    for onode in osubgraph.boundary_nodes:
      if onode not in self.subgraph.boundary_nodes and \
          self.subgraph.is_member(onode, self.graph):
        return False

    return True

  def union(self, osubgraph):
    """
    Computes the subgraph formed of the union between my subgraph and osubgraph.
    (Algorithm 1 in 3.1 in the ACL 2013 paper.)
    """
    nboundary_nodes = set()
    nboundary_edges = dict()

    for node in self.subgraph.boundary_nodes:
      edges = self.subgraph.boundary_edges[node] | \
          osubgraph.boundary_edges[node]
      if node not in osubgraph.boundary_nodes or \
          edges != self.graph.star(node): # graph.star returns all adjacent edges
        nboundary_nodes.add(node)
        nboundary_edges[node] = frozenset(edges)

    for onode in osubgraph.boundary_nodes:
      if onode not in self.subgraph.boundary_nodes:
        nboundary_nodes.add(onode)
        nboundary_edges[onode] = \
            frozenset(self.subgraph.boundary_edges[onode] | \
            osubgraph.boundary_edges[onode])
    return BoundarySubgraph(nboundary_nodes, nboundary_edges, self.subgraph.size +
        osubgraph.size)

  ## BEGIN BOUNDARY NODE REPR

  # overrides the necessary pieces of the item class

  def init_subgraph(self):
    return BoundarySubgraph(frozenset(), dict(), 0)

  def matches_whole_graph(self):
    return self.subgraph.size == len(self.graph.triples())

  def check_subgraph_overlap(self, oitem):

    if not self.is_disjoint(oitem.subgraph):
      return None
    
    return self.union(oitem.subgraph)

  def check_edge_overlap(self, edge):
    enodes = set() 
   
    # TODO: edge needs to be stripped for node labels before being added to the subgraph 
    if self.nodelabels:
        enodes.add(edge[0][0])
        for n, label in edge[2]:
            if self.graph.star(n) - set([edge]): 
                enodes.add(n)
    else:
        enodes.add(edge[0])
        for n in edge[2]:
            if self.graph.star(n) - set([edge]): 
                enodes.add(n)

    eedge = dict((n, frozenset([edge])) for n in enodes)
    esubgraph = BoundarySubgraph(enodes, eedge, 1)
    if not self.is_disjoint(esubgraph):
      return None
    return self.union(esubgraph)

  ## END


class FasterCheckBoundaryItem(BoundaryItem):
  """
  This Item class replaces the disjointness check in BoundaryItem 
  with the faster check described in section 3.4 of the ACL 2013 paper.
  """
 
  # overrides the disjointness check of BoundaryItem
  def is_disjoint(self, osubgraph):
    """
    The check from section 3.4 of the paper. As marker node we use the
    designated root node fo the input graph. 
    """
    # check that I and J have no boundary edges in common
    myedges = set()
    for edges in self.subgraph.boundary_edges.values():
        myedges.update(edges)
    oedges = set()
    for edges in osubgraph.boundary_edges.values():
        oedges.update(edges)
    if len(myedges & oedges) != 0:
        return False 
    
    # If m belongs to both I and J it must be a boundary node
    # of both. 
    marker = self.graph.roots[0]
    if self.subgraph.is_member(marker, self.graph) and \
       osubgraph.is_member(marker, self.graph) and (
           (marker not in self.subgraph.boundary_nodes) or
           (marker not in osubgraph.boundary_nodes)):
               return False
           
    return True
