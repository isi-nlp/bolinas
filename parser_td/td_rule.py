from lib.hgraph.hgraph import Hgraph, NonterminalLabel
from lib.hgraph.new_graph_description_parser import ParserError, LexerError
from lib.tree import Tree
from lib import log
from lib.exceptions import InputFormatException, BinarizationException 
#from lib import util
import sys
import cPickle as pickle
from collections import defaultdict as ddict
import re
import itertools
import tempfile
import os
import subprocess

class Rule:
  """
  A hyperedge replacement grammar rule. In addition to storing the necessary
  indexing information and graph fragment, computes various useful structures
  (like the tree decomposition) which are independent of the input graph.
  """

  @classmethod
  def load_from_file(cls, prefix):
    """
    Loads a HRG grammar from the given prefix. First attempts to load a pickled
    representation of the grammar, then falls back on plaintext.
    """

    try:
      pickle_file = open('%s.alt.pickle' % prefix)
      output = pickle.load(pickle_file)
      pickle_file.close()
      return output
    except IOError:
      log.warn('No pickled grammar---loading from string instead. ' + \
               'This could take a while.')

    lhs_file = open('%s.lhs' % prefix)
    rhs_amr_file = open('%s.rhs-amr' % prefix)

    output = {}
    while True:
      lhs = lhs_file.readline().strip()
      if not lhs:
        break
      rhs_amr = rhs_amr_file.readline().strip()

      parts = lhs.split(',', 2)
      if len(parts) == 3:
        rule_id, symbol, weight = parts
      elif len(parts) == 2:
        rule_id, symbol = parts
        weight = 1.0
      else:
        raise InputFormatException, "Invalid rule LHS: %s" % lhs

      rid2, amr_str = rhs_amr.split(',', 1)
      try:
        assert rule_id == rid2
      except AssertionError,e:
        raise InputFormatException, "Rule ID mismatch in grammar specification."
      rule_id = int(rule_id)
      if symbol[0] == '#':
        symbol = symbol[1:]
      weight = float(weight)
      try:
        amr = Amr.from_string(amr_str)
      except (ParserError, LexerError), e:
        raise InputFormatException, "Invalid graph RHS: %s" % amr_str

      assert rule_id not in output
      rule = Rule(rule_id, symbol, weight, amr)
      output[rule_id] = rule

    found_root = False
    for rule_id in output:
      if "root" in output[rule_id].symbol.lower():
        found_root = True
        break
    if not found_root:
      raise InputFormatException, "Need at least one rule with start symbol 'root' on LHS."

    lhs_file.close()
    rhs_amr_file.close()

    return output

  def __init__(self, rule_id, symbol, weight, amr):
    """
    Initializes this rule with the given id, symbol, etc.
    Next, computes an (approximate!) "nice" tree decomposition of the rule
    graph, and assembles the lookup tables which tell each tree node where to
    find its sibling and parent.
    """
    self.rule_id = rule_id
    self.symbol = symbol
    self.weight = weight

    assert len(amr.roots) == 1
    self.root_node = amr.roots[0]
    self.boundary_nodes = amr.external_nodes

    # These tables record the structure of the tree decomposition (rather than
    # making every item carry around a copy of a complete subtree). See the item
    # class for a description of how they are used.
    self.tree_to_sibling = {}
    self.tree_to_parent = {}
    self.tree_to_edge = {}
    self.tree_leaves = set()
    self.tree_second_siblings = set()
    self.tree_to_boundary_nodes = {}
    tree = tree_decomposition(amr)
    for node in tree.nodes():
      self.tree_to_boundary_nodes[node.node_id] = node.graph_nodes
      if node.first_child:
        self.tree_to_parent[node.first_child.node_id] = node.node_id
        if node.second_child:
          self.tree_to_sibling[node.first_child.node_id] = node.second_child.node_id
          self.tree_second_siblings.add(node.second_child.node_id)
      else:
        self.tree_leaves.add(node.node_id)
      if node.graph_edge:
        self.tree_to_edge[node.node_id] = node.graph_edge

    for node in tree.nodes():
      assert not (node in self.tree_to_sibling and node in self.tree_to_edge)
      assert (node in self.tree_to_sibling) or (node in self.tree_to_edge) or \
             (node in self.tree_leaves) or (node not in self.tree_to_parent)

  def __eq__(self, other):
    return isinstance(other, Rule) and other.rule_id == self.rule_id

# BEGIN TREE DECOMP

# The (ugly) code that follows computes a nice tree decomposition for the rule's
# graph. The decomposition computed here uses the heuristic described in the
# paper, and provides no formal guarantees for treewidth. 

class TreeNode:
  def __init__(self):
    self.graph_nodes = set()
    self.graph_edge = None
    self.first_child = None
    self.second_child = None

  def __repr__(self):
    return '( {%s} %s %s )' % (','.join(self.graph_nodes),
        repr(self.first_child), repr(self.second_child))

  def has_descendant(self, desc_graph_node):
    if desc_graph_node in self.graph_nodes:
      return True
    if self.first_child and self.first_child.has_descendant(desc_graph_node):
      return True
    if self.second_child and self.second_child.has_descendant(desc_graph_node):
      return True
    return False

  def add_path_from_self(self, to_graph_node):
    self.graph_nodes.add(to_graph_node)
    if self.first_child and self.first_child.has_descendant(to_graph_node):
      self.first_child.add_path_from_self(to_graph_node)
    if self.second_child and self.second_child.has_descendant(to_graph_node):
      self.second_child.add_path_from_self(to_graph_node)

  def nodes(self):
    yield self
    if self.first_child:
      for node in self.first_child.nodes():
        yield node
    if self.second_child:
      for node in self.second_child.nodes():
        yield node

  def add_running_intersection(self, amr):
    for graph_node in amr.get_nodes():
      self.add_running_intersection_single(graph_node)

  def add_running_intersection_single(self, graph_node):
    if self.second_child and \
       self.first_child.has_descendant(graph_node) and \
       self.second_child.has_descendant(graph_node) and \
       graph_node not in self.graph_nodes:
      self.add_path_from_self(graph_node)
      return

    if self.first_child:
      self.first_child.add_running_intersection_single(graph_node)
    if self.second_child:
      self.second_child.add_running_intersection_single(graph_node)

  def number(self, counter=0):
    self.node_id = counter
    counter += 1
    if self.first_child:
      counter = self.first_child.number(counter)
    if self.second_child:
      counter = self.second_child.number(counter)
    return counter

def tree_decomposition(amr):
  assert len(amr.roots) == 1
  visited = set()
  td = tree_decomposition_node(amr.roots, visited, amr)
  td.add_running_intersection(amr)
  td.number()
  return td

def tree_decomposition_node(graph_nodes, visited, amr):
  visit_edges = sum([amr.out_edges(graph_node) \
                     for graph_node in graph_nodes], [])
  if graph_nodes == amr.roots:
    visit_edges.append((amr.roots[0], 'DEPENDENCY', amr.get_external_nodes()))

  subtrees = []
  for graph_node in graph_nodes:
    subtrees += [tree_decomposition_edge(graph_edge, visited, amr) \
                 for graph_edge in amr.out_edges(graph_node) \
                 if graph_edge not in visited]

  if not subtrees:
    return TreeNode()

  while len(subtrees) > 1:
    left = subtrees.pop()
    right = subtrees.pop()
    tree_node = TreeNode()
    tree_node.first_child = left
    tree_node.second_child = right
    subtrees.append(tree_node)

  return subtrees[0]

def tree_decomposition_edge(graph_edge, visited, amr):
  visited.add(graph_edge)
  tree_node = TreeNode()
  tree_node.graph_nodes.add(graph_edge[0])
  tree_node.graph_nodes |= set(graph_edge[2])
  tree_node.graph_edge = graph_edge
  tree_node.first_child = tree_decomposition_node(graph_edge[2], visited, amr)
  return tree_node

# END

