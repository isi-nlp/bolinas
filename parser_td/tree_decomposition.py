# The code that follows computes a nice tree decomposition for the rule's
# graph. The decomposition computed here uses the heuristic described in the
# ACL 2013 paper (DFS and augmenting nodes to satisfy running intersection property)
# and provides no formal guarantees for treewidth. 

class TreeNode:
  def __init__(self):
    self.graph_nodes = set()
    self.graph_edge = None
    self.first_child = None
    self.second_child = None

  def __repr__(self):
    return '( {%s} %s %s %s )' % (','.join(self.graph_nodes), self.graph_edge,
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


if __name__ == "__main__":
    from lib.hgraph.hgraph import Hgraph
    graph = Hgraph.from_string("(n :P$1 :arg0 (a.n :E$2) :arg1 (n :S$3 a.))")
    td = tree_decomposition(graph)
