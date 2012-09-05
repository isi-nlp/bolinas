from lib.amr.dag import Dag, NonterminalLabel
from nltk.tree import Tree
import lib.log
import cPickle as pickle
from collections import defaultdict as ddict
from lib.exceptions import BinarizationException
import re

class Rule:

  @classmethod
  def load_from_file(cls, prefix):

    try:
      pickle_file = open('%s.pickle' % prefix)
      output = pickle.load(pickle_file)
      pickle_file.close()
      return output
    except IOError:
      log.warn('No pickled grammar. Loading from string instead.')

    lhs_file = open('%s.lhs' % prefix)
    rhs_amr_file = open('%s.rhs-amr' % prefix)
    rhs_ptb_file = open('%s.rhs-ptb' % prefix)

    output = {}

    while True:
      lhs = lhs_file.readline().strip()
      if not lhs:
        break
      rhs_amr = rhs_amr_file.readline().strip()
      rhs_ptb = rhs_ptb_file.readline().strip()

      rule_id, symbol, weight = lhs.split(',', 2)
      rid2, amr_str = rhs_amr.split(',', 1)
      rid3, ptb_str = rhs_ptb.split(',', 1)

      assert rule_id == rid2 == rid3
      rule_id = int(rule_id)
      symbol = symbol[1:]
      weight = float(weight)

      amr = Dag.from_string(amr_str)
      ptb = Tree(ptb_str)

      assert rule_id not in output
      rule = Rule(rule_id, symbol, weight, amr, ptb)
      output[rule_id] = rule

    lhs_file.close()
    rhs_amr_file.close()
    rhs_ptb_file.close()

    return output

  @classmethod
  def write_to_file(cls, grammar, prefix):
    pickle_file = open('%s.pickle' % prefix, 'w')
    lhs_file = open('%s.lhs' % prefix, 'w')
    rhs_amr_file = open('%s.rhs-amr' % prefix, 'w')
    rhs_ptb_file = open('%s.rhs-ptb' % prefix, 'w')
    tiburon_file = open('%s.str2amr.cfg' % prefix, 'w')

    #print >>tiburon_file, '#ROOT_SBARQ_1'
    print >>tiburon_file, """#START
#START.ROOT_SBARQ_1: -> #ROOT_SBARQ_1.ROOT_SBARQ_1 # 1
#START.ROOT_SQ_1: -> #ROOT_SQ_1.ROOT_SQ_1 # 1
#START.ROOT_SBAR_1: -> #ROOT_SBAR_1.ROOT_SBAR_1 # 1
#START.ROOT_S_1: -> #ROOT_S_1.ROOT_S_1 # 1"""

    for rule in grammar.values():
      print >>lhs_file, '%d,%s,%f' % (rule.rule_id, rule.symbol, rule.weight)
      print >>rhs_amr_file, '%d,%s' % (rule.rule_id, 
          re.sub(r'\s+', ' ', str(rule.amr)))
          #' '.join(str(rule.amr).split('\n')))
      print >>rhs_ptb_file, '%d,%s' % (rule.rule_id, 
          ' '.join(str(rule.parse).split('\n')))
      print >>tiburon_file, rule.tiburon_str()

    pickle.dump(grammar, pickle_file)
    lhs_file.close()
    rhs_amr_file.close()
    rhs_ptb_file.close()
    pickle_file.close()

  @classmethod
  def normalize_weights(cls, grammar):
    norms = ddict(lambda:0.0)
    for rule in grammar.values():
      norms[rule.symbol] += rule.weight
    ngrammar = {}
    for rule_id, rule in grammar.items():
      nrule = rule.reweight(rule.weight / norms[rule.symbol])
      ngrammar[rule_id] = nrule
    return ngrammar

  def __init__(self, rule_id, symbol, weight, amr, parse, amr_visit_order =
      None, string_visit_order = None, original_index = None):
    assert len(amr.roots) == 1
    self.rule_id = rule_id
    self.symbol = symbol
    self.weight = weight
    self.amr = amr
    self.parse = parse
    if isinstance(parse, Tree):
      self.string = parse.leaves()
    else:
      self.string = [parse]
    #self.string = parse.leaves() if isinstance(parse, Tree) else [parse]

    if amr_visit_order == None:
      self.amr_visit_order = range(len(amr.triples()))
    else:
      self.amr_visit_order = amr_visit_order

    if string_visit_order == None:
      self.string_visit_order = range(len(self.string))
    else:
      self.string_visit_order = string_visit_order

    if original_index != None:
      self.original_index = original_index

    self.is_terminal = not any(w[0] == '#' for w in self.string)

  def reweight(self, nweight):
    return Rule(self.rule_id, self.symbol, nweight, self.amr, self.parse, \
        self.amr_visit_order, self.string_visit_order)

  def canonicalize_amr(self):
    return Rule(self.rule_id, self.symbol, self.weight,
        self.amr.clone_canonical(), self.parse, self.amr_visit_order,
        self.string_visit_order)

  def __repr__(self):
    return 'Rule(%d,%s)' % (self.rule_id, self.symbol)

  def __hash__(self):
    return self.rule_id

  def __eq__(self, other):
    return isinstance(other, Rule) and self.rule_id == other.rule_id

  def terminal_search(self, root, triples, allow_backward):
    #print 'search from', root
    #print 'in', triples
    stack = []
    for r in root[2]:
      stack.append(r)
    if allow_backward:
      stack.append(root[0])

    out = []

    while stack:
      top = stack.pop()
      children = [t for t in triples if t[0] == top and not isinstance(t[1],
        NonterminalLabel) and t not in out]
      if allow_backward:
        children += [t for t in triples if top in t[2] and not isinstance(t[1],
          NonterminalLabel) and t not in out]
      #print 'backward?', allow_backward
      #print 'children of', top, ':', children
      for c in children:
        out.append(c)
        for t in c[2]:
          stack.append(t)
        if allow_backward:
          stack.append(c[0])

    return out

  #def make_rule(*args):
  #  if BINARIZE_STRING:
  #    return args[0].make_string_rule(*args[1:])
  #  else:
  #    return args[0].make_tree_rule(*args[1:])

  def make_string_rule(self, string, amr, rule_string, rule_amr, next_id):

    new_rule_id = next_id + 1
    #new_symbol = '%d__%d[%d]' % (self.rule_id, new_rule_id, new_rule_id)
    new_symbol = '%d__%d' % (self.rule_id, new_rule_id)
    rule_tree = Tree('X', rule_string)

    amr_t_indices = []
    amr_nt_indices = []
    for i in range(len(rule_amr.triples())):
      if isinstance(rule_amr.triples()[i][1], NonterminalLabel):
        amr_nt_indices.append(i)
      else:
        amr_t_indices.append(i)
    amr_visit_order = amr_t_indices + amr_nt_indices

    string_t_indices = []
    string_nt_indices = []
    for i in range(len(rule_string)):
      if rule_string[i][0] == '#':
        string_nt_indices.append(i)
      else:
        string_t_indices.append(i)
    #string_visit_order = string_t_indices + string_nt_indices
    # can't handle spanning
    string_visit_order = range(len(rule_string))

    if len(amr_t_indices) == 0  and len(string_t_indices) == 0:
      assert string_visit_order == [0,1]
      assert amr_visit_order == [0,1]
      if rule_string[0] != str(rule_amr.triples()[0][1]):
        amr_visit_order = [1,0]

    external = []
    for node in rule_amr:
      if node in amr.external_nodes:
        external.append(node)
      adjacent_edges = amr.in_edges(node) + amr.out_edges(node)
      if any(e not in rule_amr.triples() for e in adjacent_edges):
        external.append(node)
    external = list(set(external))
    rule_root = list(rule_amr.roots)[0]
    if rule_root in external:
      external.remove(rule_root)
    if len(external) == 0:
      external.append(rule_amr.root_edges()[0][2][0])
    rule_amr.external_nodes = external

    new_rule = Rule(new_rule_id, new_symbol, 1, rule_amr, rule_tree,
        amr_visit_order, string_visit_order)

    o_amr = amr.collapse_fragment(rule_amr, NonterminalLabel(new_symbol))
    #string_cut_start = string.index(rule_string[0])
    #string_cut_end = string.index(rule_string[-1]) + 1
    string_cut_indices = [i for i in range(len(string)) if \
        string[i:i+len(rule_string)] == rule_string]
    assert len(string_cut_indices) == 1
    string_cut_start = string_cut_indices[0]
    string_cut_end = string_cut_start + len(rule_string)
    o_string = list(string)
    o_string[string_cut_start:string_cut_end] = ['#%s' % new_symbol]

    if len(o_amr.external_nodes) == 0:
      o_amr.external_nodes.append(o_amr.root_edges()[0][2][0])

    return new_rule, o_string, o_amr, next_id+1

  def collapse_constituent(self, tree, constituent, label):
    if tree == constituent:
      return str(label)
    if not isinstance(tree, Tree):
      return tree
    n_tree = Tree(tree.node, [self.collapse_constituent(subtree, constituent,
      label) for subtree in tree])
    return n_tree

  def make_tree_rule(self, tree, amr, rule_tree, rule_amr, next_id):

    new_rule_id = next_id + 1
    new_symbol = '%d__%d' % (self.rule_id, new_rule_id)

    if isinstance(rule_tree, Tree):
      rule_string = rule_tree.leaves()
    else:
      rule_string = [rule_tree]

    amr_t_indices = []
    amr_nt_indices = []
    for i in range(len(rule_amr.triples())):
      if isinstance(rule_amr.triples()[i][1], NonterminalLabel):
        amr_nt_indices.append(i)
      else:
        amr_t_indices.append(i)
    amr_visit_order = amr_t_indices + amr_nt_indices

    string_visit_order = range(len(rule_string))
    string_t_indices = [s for s in rule_string if s[0] == '#']

    if len(amr_t_indices) == 0 and len(string_t_indices) == 0:
      assert string_visit_order == [0,1]
      assert amr_visit_order == [0,1]
      if rule_string[0] != str(rule_amr.triples()[0][1]):
        amr_visit_order = [1,0]

    external = []
    for node in rule_amr:
      if node in amr.external_nodes:
        external.append(node)
      adjacent_edges = amr.in_edges(node) + amr.out_edges(node)
      if any(e not in rule_amr.triples() for e in adjacent_edges):
        external.append(node)
    external = list(set(external))
    rule_root = list(rule_amr.roots)[0]
    if rule_root in external:
      external.remove(rule_root)
    if len(external) == 0:
      external.append(rule_amr.root_edges()[0][2][0])
    rule_amr.external_nodes = external

    new_rule = Rule(new_rule_id, new_symbol, 1, rule_amr, rule_tree,
        amr_visit_order, string_visit_order)

    label = NonterminalLabel(new_symbol)
    o_amr = amr.collapse_fragment(rule_amr, label)
    o_tree = self.collapse_constituent(tree, rule_tree, label)

    if len(o_amr.external_nodes) == 0:
      o_amr.external_nodes.append(o_amr.find_leaves()[0])

    return new_rule, o_tree, o_amr, next_id+1

  def collapse_amr_terminals(self, string, amr, next_id):
    nonterminals = list(reversed([t for t in amr.triples() if isinstance(t[1],
      NonterminalLabel)]))

    rules = []

    first = True
    while nonterminals:
      nt = nonterminals.pop()
      attached_terminals = self.terminal_search(nt, amr.triples(), first)
      first = False
      if not attached_terminals:
        continue

      #print 'triples', [nt] + attached_terminals

      rule_amr = Dag.from_triples([nt] + attached_terminals)
      rule_string = [str(nt[1])]

      if len(rule_amr.roots) > 1:
        # TODO actually I just need to fix the search impl
        raise BinarizationException

      new_rule, string, amr, next_id = self.make_string_rule(string, amr,
          rule_string, rule_amr, next_id)
      rules.append(new_rule)

    return string, amr, rules, next_id

  def collapse_amr_terminals_tree(self, tree, amr, next_id):
    nonterminals = list(reversed([t for t in amr.triples() if isinstance(t[1],
      NonterminalLabel)]))

    rules = []

    first = True
    while nonterminals:
      nt = nonterminals.pop()
      attached_terminals = self.terminal_search(nt, amr.triples(), first)
      first = False
      if not attached_terminals:
        continue

      #print 'triples', [nt] + attached_terminals

      rule_amr = Dag.from_triples([nt] + attached_terminals)
      rule_tree = str(nt[1])

      if len(rule_amr.roots) > 1:
        # TODO actually I just need to fix the search impl
        raise BinarizationException

      new_rule, tree, amr, next_id = self.make_tree_rule(tree, amr,
          rule_tree, rule_amr, next_id)
      rules.append(new_rule)

    return tree, amr, rules, next_id

  def collapse_string_terminals(self, string, amr, next_id):
    nonterminals = list(reversed([t for t in string if t[0] == '#']))

    rules = []

    slice_from = 0

    while nonterminals:
      nt = nonterminals.pop()
      if nonterminals:
        slice_to = string.index(nonterminals[-1])
      else:
        slice_to = len(string)

      if slice_to - slice_from == 1:
        slice_from = slice_to
        continue

      rule_string = string[slice_from:slice_to]
      nt_edge_l = [e for e in amr.triples() if str(e[1]) == nt]
      assert len(nt_edge_l) == 1
      rule_amr = Dag.from_triples(nt_edge_l)

      new_rule, string, amr, next_id = self.make_string_rule(string, amr, rule_string,
          rule_amr, next_id)
      rules.append(new_rule)
      
      slice_from = slice_from + 1 # = slice_to - (slice_to - slice_from) + 1

    return string, amr, rules, next_id

  def merge_nonterminals_tree(self, tree, amr, next_id):

    rules = []

    while True:
      if not isinstance(tree, Tree):
        assert len(amr.triples()) == 1
        return tree, amr, rules, next_id

      # a collapsible subtree consists of
      # 1. many terminals
      # 2. one nonterminal and many terminals
      # 3. two nonterminals
      collapsible_subtrees = []
      for st in tree.subtrees():
        terminals = [t for t in st.leaves() if t[0] == '#']
        if len(terminals) == 1:
          collapsible_subtrees.append(st)
        elif len(terminals) == 2 and len(st.leaves()) == 2:
          collapsible_subtrees.append(st)

      rule_tree = max(collapsible_subtrees, key=lambda x: x.height())
      terminals = [t for t in rule_tree.leaves() if t[0] == '#']
      rule_edge_l = [t for t in amr.triples() if str(t[1]) in terminals]
      rule_amr = Dag.from_triples(rule_edge_l)
      if len(rule_amr.roots) != 1:
        raise BinarizationException

      new_rule, tree, amr, next_id = self.make_tree_rule(tree, amr, rule_tree,
          rule_amr, next_id)
      rules.append(new_rule)

    return tree, amr, rules, next_id

  def merge_nonterminals(self, string, amr, next_id):
    rules = []

    stack = []
    tokens = list(reversed([s for s in string if s]))

    while tokens:
      next_tok = tokens.pop()
      
      #print next_tok
      #print amr.triples()
      #print str(amr.triples()[0][1])
      #print [t for t in amr.triples() if str(t[1]) == next_tok]
      next_tok_triple_l = [t for t in amr.triples() if str(t[1]) == next_tok]
      next_tok_triple = [t for t in amr.triples() if str(t[1]) == next_tok][0]
      if not stack:
        stack.append(next_tok)
        continue
      stack_top = stack.pop()
      stack_top_triple = [t for t in amr.triples() if str(t[1]) == stack_top][0]

      #stack_top_nodes = set((stack_top_triple[0],) + stack_top_triple[2])
      #next_tok_nodes = set((next_tok_triple[0],) + next_tok_triple[2])

      #if len(stack_top_nodes & next_tok_nodes) == 0:
      if (stack_top_triple[0] not in next_tok_triple[2]) and \
         (next_tok_triple[0] not in stack_top_triple[2]):
        stack.append(stack_top)
        stack.append(next_tok)
        continue

      rule_amr = Dag.from_triples([stack_top_triple, next_tok_triple])

      if len(rule_amr.roots) > 1:
        assert False

      rule_string = [stack_top, next_tok]
      new_rule, string, amr, next_id = self.make_string_rule(string, amr, rule_string,
          rule_amr, next_id)
      tokens.append('#%s' % new_rule.symbol)

      rules.append(new_rule)

    if len(stack) > 1:
      raise BinarizationException
      #print 'unbinarizable'
      #print self.amr
      #print self.string
      #assert False

    return string, amr, rules, next_id

  def binarize_string(self, next_id):
    oid = next_id

    string = self.string
    amr = self.amr

    # handle all-terminal rules
    if not any(s[0] == '#' for s in string):
      return [Rule(next_id, self.symbol, self.weight, self.amr, self.parse,
          self.amr_visit_order, self.string_visit_order)], next_id + 1

    rules = []

    try:
      # remove all nonterminal edges from amr
      string, amr, at_rules, next_id = self.collapse_amr_terminals(string, amr,
          next_id)
      rules += at_rules

      string, amr, st_rules, next_id = self.collapse_string_terminals(string, amr,
          next_id)
      rules += st_rules

      string, amr, nt_rules, next_id = self.merge_nonterminals(string, amr,
          next_id) 
      rules += nt_rules
    except BinarizationException:
      log.warn('Unbinarizable rule!')
      return None, oid

    assert len(string) == 1
    assert len(amr.triples()) == 1
    rules.append(Rule(next_id + 1, self.symbol, self.weight, amr, Tree('X',
      string)))

    return rules, next_id + 2

  def max_collapsable_tree(self, nt, tree):
    # TODO this could be more efficient
    collapsable_trees = [st for st in tree.subtrees() if nt in st.leaves() and
        len([l for l in st.leaves() if l[0] == '#']) == 1]
    if len(collapsable_trees) == 0:
      return nt
    else:
      return max(collapsable_trees, key=lambda x: len(x.leaves()))

  def collapse_tree_terminals(self, tree, amr, next_id):
    if isinstance(tree, Tree):
      string = tree.leaves()
    else:
      string = [tree]
    nonterminals = list(reversed([t for t in string if t[0] == '#']))

    rules = []

    while nonterminals:

      nt = nonterminals.pop()
      rule_tree = self.max_collapsable_tree(nt, tree)

      nt_edge_l = [e for e in amr.triples() if str(e[1]) == nt]
      assert len(nt_edge_l) == 1
      rule_amr = Dag.from_triples(nt_edge_l)

      new_rule, tree, amr, next_id = self.make_tree_rule(tree, amr,
          rule_tree, rule_amr, next_id)
      
      if isinstance(tree, Tree):
        string = tree.leaves()
      else:
        string = [tree]

    return tree, amr, rules, next_id

  def binarize_tree(self, next_id):
    oid = next_id

    tree = self.parse
    amr = self.amr

    if not any(s[0] == '#' for s in self.string):
      return [Rule(next_id, self.symbol, self.weight, self.amr, self.parse,
        self.amr_visit_order, self.string_visit_order)], next_id + 1

    rules = []

    try:
      tree, amr, nt_rules, next_id = self.collapse_amr_terminals_tree(tree, amr,
          next_id)

      tree, amr, nt_rules, next_id = self.merge_nonterminals_tree(tree, amr,
          next_id)
      rules += nt_rules
    except BinarizationException:
      log.warn('Unbinarizable rule!')
      return None, oid

    assert isinstance(tree, str)
    assert len(amr.triples()) == 1
    rules.append(Rule(next_id + 1, self.symbol, self.weight, amr, tree))

    return rules, next_id + 2

  def binarize(self, next_id, BINARIZE_STRING):
    if BINARIZE_STRING:
      return self.binarize_string(next_id)
    else:
      return self.binarize_tree(next_id)

  def tiburon_str(self):
    d_tree = ' '.join(['%s:' % s[1:] for s in self.string if s[0] == '#'])
    d_string = ' '.join(['%s.%s' % (s.split('[')[0], s[1:]) if s[0] == '#' else s for s in
      self.string])
    if d_tree:
      return '#%s.%s(%d(%s)) -> %s # %f' % (self.symbol, self.symbol,
          self.rule_id, d_tree, d_string, self.weight)
    else:
      return '#%s.%s(%d) -> %s # %f' % (self.symbol, self.symbol, self.rule_id,
          d_string, self.weight)
