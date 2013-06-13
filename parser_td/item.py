from lib.amr.dag import Dag, NonterminalLabel
from lib import log
import sys
from collections import defaultdict as ddict

class Subgraph:
  """
  Represents a boundary-induced subgraph, for fast union computation.
  """
  def __init__(self, boundary_nodes, boundary_edges, size):
    self.boundary_nodes = frozenset(boundary_nodes)
    self.boundary_edges = ddict(frozenset, boundary_edges)
    self.size = size

    self.hedges = frozenset([(k,boundary_edges[k]) for k in boundary_edges])
    self.saved_hash = 17 * hash(self.boundary_nodes) + 23 * hash(self.hedges)

  def __hash__(self):
    return self.saved_hash

  def __eq__(self, other):
    return isinstance(other, Subgraph) and \
        self.boundary_nodes == other.boundary_nodes and \
        self.hedges == other.hedges

  def is_member(self, node, graph):
    """
    Tests if node is a member of this subgraph.
    (Algorithm 1 in the paper.)
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


class Item:
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

  def __init__(self, rule, tree_node, graph, subgraph=None, mapping=None):
    self.rule = rule
    self.tree_node = tree_node
    self.graph = graph

    # tree_node specifies the subtree of the tree decomposition which has
    # *already* been recognized

    # Every node of the tree decomposition will be one of
    # 1. A leaf
    # 2. A node introducing a single edge
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
    return '[%d, %s, %d, {%d}]' % (self.rule.rule_id, self.rule.symbol, \
        self.tree_node, len(self.subgraph))

  ## BEGIN EXPLICIT REPR

  # these methods can be overriden to use a more compact subgraph representation
  # (e.g. boundary nodes)

  def init_subgraph(self):
    """
    Produces the object representing an empty subgraph.
    """
    return frozenset()

  def matches_whole_graph(self):
    """
    Determines whether this item has recognized the entire input graph.
    """
    return len(self.subgraph) == len(self.graph.triples())

  def check_subgraph_overlap(self, oitem):
    """
    Determines whether this item and oitem recognize non-disjoint subgraphs.
    """
    for edge in oitem.subgraph:
      if edge in self.subgraph:
        return None
    nsubgraph = frozenset(self.subgraph | oitem.subgraph)
    return nsubgraph

  def check_edge_overlap(self, edge):
    """
    Determines whether edge overlaps with this item's recognized subgraph.
    """
    if edge in self.subgraph:
      return None
    nsubgraph = frozenset(self.subgraph | set([edge]))
    return nsubgraph

  ## END

  def check_mapping_bijection_nonterminal(self, oitem):
    """
    Determines whether this item's node mapping and oitem's, when merged, still
    form a valid bijection. (oitem is passive).  If so, returns the resulting
    map.
    """
    # create an empty mapping
    nmapping = dict(self.mapping)

    # oitem is a passive item with a different rule, so the node names in its
    # mapping are totally irrelevant
    # we're matching it against one edge in this rule, so we need to make sure
    # that the head and tails line up properly

    # pull out the roots of oitem, and the piece of my rule that it corresponds
    # to
    # myhead is NOT the root of my rule! it's the head node of the edge I'm
    # matching
    myhead = self.next_key_edge[0]
    oroot = oitem.rule.root_node

    # if oitem has mapped its root, and I have mapped the corresponding head
    # node in my rule, make sure the two mappings agree
    graph_root = oitem.mapping[oroot]
    if myhead in self.mapping and self.mapping[myhead] != graph_root:
      return None
    nmapping[myhead] = graph_root

    # now repeat that procedure for every tail node
    for i in range(len(self.next_key_edge[2])):
      mynode = self.next_key_edge[2][i]
      onode = oitem.rule.boundary_nodes[i]
      graph_node = oitem.mapping[onode]
      if mynode in self.mapping and self.mapping[mynode] != graph_node:
        return None
      nmapping[mynode] = graph_node

    return nmapping

  def check_mapping_bijection_binary(self, oitem):
    """
    As in check_mapping_bijection_nonterminal, but for active oitem.
    """
    # unlike above, oitem has the same rule (and so the same node names) that I
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
    return nmapping

  def check_mapping_bijection_terminal(self, edge):
    """
    As in the previous method, but with a terminal edge rather than a chart
    item.
    """
    # this time there's only one edge to worry about, so I just need to check
    # for internal consistency
    if self.next_key_edge[0] in self.mapping:
      if self.mapping[self.next_key_edge[0]] != edge[0]:
        return None
    nmapping = dict(self.mapping)
    nmapping[self.next_key_edge[0]] = edge[0]

    for i in range(len(self.next_key_edge[2])):
      if self.next_key_edge[2][i] in self.mapping:
        if self.mapping[self.next_key_edge[2][i]] != edge[2][i]:
          return None
      nmapping[self.next_key_edge[2][i]] = edge[2][i]
    return nmapping

  def terminal(self, edge):
    """
    Attempts to apply the Terminal rule and consume edge. Returns the resulting
    item, if the attempt succeeded.
    """
    if self.target != Item.TERMINAL:
      return None
    if edge[1] != self.next_key:
      return None
    if len(edge[2]) != len(self.next_key_edge[2]):
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
        nmapping)

  def nonterminal(self, oitem):
    """
    Attempts to apply the Nonterminal rule and consume oitem, returning the
    result if successful.
    """
    if self.target != Item.NONTERMINAL:
      #print 'I am not NT'
      return None
    if oitem.target != Item.ROOT:
      #print 'other is not root'
      return None
    if oitem.rule.symbol != self.next_key:
      #print 'symbol mismatch'
      return None
    oboundary = oitem.rule.tree_to_boundary_nodes[oitem.tree_node]
    if len(oboundary) != len(self.next_key_edge[2]) + 1:
      #print 'boundary mismatch'
      return None

    nsubgraph = self.check_subgraph_overlap(oitem)
    if not nsubgraph:
      #print 'overlap'
      return None
    nmapping = self.check_mapping_bijection_nonterminal(oitem)
    if not nmapping:
      #print 'bijection'
      return None

    return self.__class__(self.rule,
        self.rule.tree_to_parent[self.tree_node],
        self.graph,
        nsubgraph,
        nmapping)

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
        nmapping)


class BoundaryItem(Item):
  """
  A drop-in replacement for the Item class which uses a boundary representation
  rather than a list of all the recognized edges.
  """

  def is_disjoint(self, osubgraph):
    """
    Checks whether my subgraph and osubgraph are disjoint.
    (Algorithm 2 in the paper.)
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
    (Algorithm 3 in the paper.)
    """
    nboundary_nodes = set()
    nboundary_edges = dict()

    for node in self.subgraph.boundary_nodes:
      edges = self.subgraph.boundary_edges[node] | \
          osubgraph.boundary_edges[node]
      if node not in osubgraph.boundary_nodes or \
          edges != self.graph.star(node):
        nboundary_nodes.add(node)
        nboundary_edges[node] = frozenset(edges)

    for onode in osubgraph.boundary_nodes:
      if onode not in self.subgraph.boundary_nodes:
        nboundary_nodes.add(onode)
        nboundary_edges[onode] = \
            frozenset(self.subgraph.boundary_edges[onode] | \
            osubgraph.boundary_edges[onode])

    return Subgraph(nboundary_nodes, nboundary_edges, self.subgraph.size +
        osubgraph.size)

  ## BEGIN EXPLICIT REPR

  # overrides the necessary pieces of the item class

  def init_subgraph(self):
    return Subgraph(frozenset(), dict(), 0)

  def matches_whole_graph(self):
    return self.subgraph.size == len(self.graph.triples())
    print self.subgraph.size

  def check_subgraph_overlap(self, oitem):
    if not self.is_disjoint(oitem.subgraph):
      return None
    return self.union(oitem.subgraph)

  def check_edge_overlap(self, edge):
    enodes = set([edge[0]] + list(edge[2]))
    eedge = dict((n, frozenset([edge])) for n in enodes)
    esubgraph = Subgraph(enodes, eedge, 1)
    if not self.is_disjoint(esubgraph):
      return None
    return self.union(esubgraph)

  ## END
