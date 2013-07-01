from common.rule import Rule
from tree_decomposition import tree_decomposition

class TdRule(Rule):
    """
    A hyperedge replacement grammar rule. In addition to storing the necessary
    indexing information and graph fragment, computes various useful structures
    (like the tree decomposition) which are independent of the input graph.
    """
  
    def __init__(self, rule_id, symbol, weight, rhs1, rhs2, nodelabels = False):
        """
        Initializes this rule with the given id, symbol, etc.
        Next, computes an (approximate!) "nice" tree decomposition of the rule
        graph, and assembles the lookup tables which tell each tree node where to
        find its sibling and parent.
        """
        self.rule_id = rule_id
        self.symbol = symbol
        self.weight = weight
            
        self.rhs1 = rhs1 # These are not needed once the TD is computed, but we use them
        self.rhs2 = rhs2 #   to reassemble output structures

        self.nodelabels = nodelabels
     
        assert len(rhs1.roots) == 1
        
        self.root_node = rhs1.roots[0]
        self.boundary_nodes = rhs1.external_nodes

        # These tables record the structure of the tree decomposition in the rule 
        # (rather than making every chart item carry around a copy of a complete subtree).
        # See the item class for a description of how they are used.
        self.tree_to_sibling = {}
        self.tree_to_parent = {}
        self.tree_to_edge = {}
        self.tree_leaves = set()
        self.tree_second_siblings = set()
        self.tree_to_boundary_nodes = {}
        tree = tree_decomposition(rhs1, nodelabels = nodelabels)
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
          return isinstance(other, TdRule) and other.rule_id == self.rule_id

