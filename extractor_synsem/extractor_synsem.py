from common.exceptions import InvocationException
from common.hgraph.hgraph import Hgraph
from common.cfg import NonterminalLabel
from lib.tree import Tree
import re
from collections import defaultdict as ddict
import itertools
from parser.vo_rule import Rule
import sys

DEFAULT_COMPOSITION_DEPTH = 3

class ExtractorSynSem:

  def __init__(self):
    pass

  @classmethod
  def help(self):
    """
    Returns SynSem help message.
    """
    return 'Usage: python extract-synsem <nl_file> <mr_file> ' + \
        '<alignment_file> <destination> [composition_depth (default %d)]' % \
        DEFAULT_COMPOSITION_DEPTH


  def main(self, *args):
    """
    Extracts rules from the given training data, with an optional composition
    depth specified.
    """
    if len(args) < 4:
      print self.help()
      raise InvocationException()
    nl_path, mr_path, align_path, destination_prefix = args[:4]
    if len(args) == 4:
      composition_depth = DEFAULT_COMPOSITION_DEPTH
    elif len(args) == 5:
      composition_depth = int(args[4])
    else:
      print self.help()
      raise InvocationException()

    self.extract_rules_corpus(nl_path, mr_path, align_path, destination_prefix,
        composition_depth)

  def extract_rules_corpus(self, nl_path, amr_path, alignment_path,
      destination_prefix, composition_depth):
    """
    Extract all rules from the corpus specified by the *_path arguments.
    """

    syn_f = open(nl_path)
    sem_f = open(amr_path)
    align_f = open(alignment_path)

    n_examples = count_lines(amr_path)
    announce_interval = n_examples / 10

    # load input data into examples list
    examples = []
    for example_i in range(n_examples):
      syn_s = syn_f.readline().strip()
      sem_s = sem_f.readline().strip()
      align_s = align_f.readline().strip()

      amr = Dag.from_string(sem_s)
      tree = Tree(syn_s)
      label_spans(tree)
      align = get_alignments(align_s, amr)

      examples.append((amr, tree, align))

    # extract rules from data
    rules = []
    for example in examples:
      example_rules = extract_rules(example[0], example[1], example[2],
          composition_depth)
      rules += example_rules

    # assign ML weights by counting
    grammar = collect_counts(rules)
    Rule.write_to_file(grammar, destination_prefix)

def count_lines(filename):
  """
  Counts the number of lines in the given file.
  """
  n_lines = 0
  with open(filename) as f:
    for line in f:
      n_lines += 1
  return n_lines

def get_alignments(align_s, amr):
  """
  Converts alignments into an actual mapping into edges of the AMR object.
  """
  alignments = ddict(list)
  align_s_parts = align_s.split()
  for part in align_s_parts:
    match = re.match(r'([^:]+):([^:]+:?[^:]+):([^:]+)-(\d+)', part)
    head = match.group(1)
    label = match.group(2)
    tail = match.group(3)
    index = int(match.group(4))

    edge_l = [e for e in amr.triples() if
                e[0] == head and \
                e[1] == label and \
                e[2] == (tail,)]
    assert len(edge_l) == 1
    alignments[edge_l[0]].append(index)
  return dict(alignments)

def label_spans(tree, start=0):
  """
  Labels each constituent with its corresponding sentence span (so that we can
  distinguish constituents over different parts of the sentence with identical
  tree structure.
  """
  end = start
  if isinstance(tree, Tree):
    for child in tree:
      end = label_spans(child, end)
    tree.span = (start, end)
    return end
  else:
    return end + 1

def minimal_aligned(constituents, tree_aligned):
  """
  Finds frontier constituents.
  """
  minimal_constituents = []
  for key in constituents:
    start,end,height = key
    # ignore unaligned constituents
    if len(tree_aligned[key]) == 0:
      continue
    # ignore constituents which have children with identical alignments
    minimal = True
    for key2 in constituents:
      start2,end2,height2 = key2
      if tree_aligned[key] == tree_aligned[key2] and start2 >= start and \
          end2 <= end and height2 < height:
        minimal = False
        break
    if not minimal:
      continue
    minimal_constituents.append(key)
  return minimal_constituents

# HERE BE DRAGONS
# The following methods implement various searches through the AMR necessary to
# produce the heuristic attachment of unaligned edges described in the paper.

def amr_reachable_h(edges, amr, predicate, expander, seen=None):
  if seen == None:
    seen = set()
  for e in edges:
    if e in seen:
      continue
    if not predicate(e):
      continue
    seen.add(e)
    amr_reachable_h(expander(e), amr, predicate, expander, seen)
  return seen

def a_parents(edge, amr):
  return amr.in_edges(edge[0])

def a_children(edge, amr):
  for t in edge[2]:
    return amr.out_edges(t)

def amr_reachable_forward(edges, amr, predicate):
  return amr_reachable_h(edges, amr, predicate, lambda e: a_parents(e, amr))

def amr_reachable_backward(edges, amr, predicate):
  return amr_reachable_h(edges, amr, predicate, lambda e: a_children(e, amr))

def amr_reachable_nothru_i(edge, amr, predicate, reachable, seen):
  if edge in seen:
    return
  seen.add(edge)
  if all(c in reachable for c in a_parents(edge, amr)):
    for c in a_parents(edge, amr):
      if all(p in reachable for p in a_children(edge, amr)):
        amr_reachable_nothru_i(c, amr, predicate, reachable, seen)
  if all(p in reachable for p in a_children(edge, amr)):
    for p in a_children(edge, amr):
      if all(c in reachable for c in a_parents(edge, amr)):
        amr_reachable_nothru_i(p, amr, predicate, reachable, seen)

def amr_reachable_nothru(edges, amr, predicate=lambda e: True):
  forward = amr_reachable_forward(edges, amr, predicate)
  backward = amr_reachable_backward(edges, amr, predicate)
  reachable = forward | backward
  seen = set()
  for edge in edges:
    amr_reachable_nothru_i(edge, amr, predicate, reachable, seen)
  return seen

def minimal_frontier(frontier):
  """
  Extracts the minimal frontier set from the given frontier set.
  """
  min_frontier = []
  for f in frontier:
    fstart, fend = f[0].span
    minimal = True
    for g in frontier:
      gstart, gend = g[0].span
      if gstart >= fstart and gend <= fend and g[0].height() < f[0].height():
        minimal = False
        break
    if minimal:
      min_frontier.append(f)
  return min_frontier

def frontier_edges(amr, tree, alignments):
  """
  Extracts the frontier set.
  """
  frontier = []
  constituents = {}
  if isinstance(tree, Tree):
    for constituent in tree.subtrees():
      key = (constituent.span[0], constituent.span[1], constituent.height())
      assert key not in constituents
      constituents[key] = constituent

  tree_aligned = ddict(set)
  for edge in alignments:
    for index in alignments[edge]:
      for key in constituents:
        start,end,height = key
        if start <= index < end:
          tree_aligned[key].add(index)

  aligned_constituents = minimal_aligned(constituents, tree_aligned)

  for key in aligned_constituents:
    start,end,height = key
    constituent = constituents[key]
    aligned_edges = [e for e in alignments if all(start <= i < end for i in
      alignments[e])]
    
    if constituent == tree:
      reachable_edges = amr.triples()
    else:
      reachable_edges = amr_reachable_nothru(aligned_edges, amr,
          lambda e: e in aligned_edges or e not in alignments)

    aligned_fragment = Dag.from_triples(reachable_edges)
    if len(aligned_fragment.root_edges()) == 1:
      frontier.append((constituent, aligned_fragment))

  min_frontier = minimal_frontier(frontier)
  min_frontier_sorted = sorted(min_frontier, key = lambda m:
      len(list(m[0].subtrees())))

  return min_frontier_sorted

def collapse_constituent(tree, constituent, label):
  """
  Shortcut: replaces a constituent with a single nonterminal label.
  """
  return replace_constituent(tree, constituent, str(label))

def replace_constituent(tree, constituent, new_constituent):
  """
  Replaces one constituent in this tree with another.
  """
  # We injected span, so the standard __eq__ check doesn't look for it
  if tree == constituent and (not isinstance(tree, Tree) or tree.span ==
      constituent.span):
    return new_constituent
  if not isinstance(tree, Tree):
    return tree
  n_tree = Tree(tree.node, [replace_constituent(subtree, constituent,
    new_constituent) for subtree in tree])
  n_tree.span = tree.span
  return n_tree

def collapse_alignments(alignments, amr_fragment, new_triple):
  """
  Adjusts alignments when replacing collapsing graph & tree fragments.
  """
  new_alignments = {}
  new_triple_alignment = []
  for triple in alignments:
    if triple in amr_fragment.triples():
      new_triple_alignment += alignments[triple]
    else:
      new_alignments[triple] = alignments[triple]
  new_triple_alignment = list(set(new_triple_alignment))
  new_alignments[new_triple] = new_triple_alignment
  return new_alignments

def make_rule(frontier_pair, amr, tree, align, next_index):
  """
  Creates a new rule with the given parts, and collapses these parts in the
  original graph and tree.
  """

  constituent, amr_fragment = frontier_pair
  outside_edges = [e for e in amr.triples() if e not in amr_fragment.triples()]

  root_label = amr_fragment.root_edges()[0][1]
  if isinstance(root_label, NonterminalLabel):
    symbol = root_label.label
    m = re.match(r'(.+)_(.+)_(\d+)', symbol)
    role = m.group(1)
  else:
    if ':' in root_label:
      role, concept = root_label.split(':')
    else:
      role = root_label

  external_nodes = amr.find_external_nodes(amr_fragment)
  if len(external_nodes) == 0:
    external_nodes = [amr_fragment.find_leaves()[0]]
  # WARNING: destructive. Unfortunately we can't make the change any earlier.
  # TODO why?
  amr_fragment.external_nodes = external_nodes

  symbol = '%s_%s_%d' % (role, constituent.node, len(external_nodes))
  label = NonterminalLabel(symbol, next_index)

  new_triple = (amr_fragment.roots[0], label, tuple(external_nodes))
  new_amr = amr.collapse_fragment(amr_fragment, label)
  assert new_triple in new_amr.triples()
  new_tree = collapse_constituent(tree, constituent, label)
  new_alignments = collapse_alignments(align, amr_fragment, new_triple)

  rule = Rule(0, symbol, 1, amr_fragment, constituent, original_index =
      next_index)

  return rule, new_amr, new_tree, new_alignments, next_index+1

def make_composed_rule(rule, cdict):
  """
  Creates a composed rule by replacing every nonterminal in this rule's RHS with
  the graph and tree fragment specified in cdict.
  """
  for label, crule in cdict.items():
    replacement_triple_l = [e for e in rule.amr.triples() if e[1] == label]
    assert len(replacement_triple_l) == 1
    replacement_fragment = Dag.from_triples(replacement_triple_l)

    new_amr = rule.amr.replace_fragment(replacement_fragment, crule.amr)
    new_tree = replace_constituent(rule.parse, str(label), crule.parse)

    new_rule = Rule(rule.rule_id, rule.symbol, rule.weight, new_amr, new_tree,
        original_index = rule.original_index)
    rule = new_rule
  return rule

def make_composed_rules(rules, max_depth):
  """
  Finds all possible composed rules, up to the specified max depth.
  """
  composed_rules = []

  # add all base rules
  for rule in rules:
    composed_rules.append(rule)

  # incrementally compose rules up to the max depth
  for i in range(1, max_depth):
    composed_rules_this_depth = []
    # consider each rule...
    for rule in rules:
      nt_labels = [e[1] for e in rule.amr.triples() if isinstance(e[1],
        NonterminalLabel)]
      if len(nt_labels) == 0:
        continue

      # ...and try to replace its nonterminals with the fragments from other
      # composed rules

      # we cheat here by relying on the fact that nonterminal indices are
      # never repeated in the induced derivation of a training example (so if a
      # rule has original_index n, we are sure it can only replace the
      # nonterminal with the same index)
      composition_candidates = {}
      for label in nt_labels:
        composition_candidates[label] = []
        for crule in composed_rules:
          if crule.original_index != label.index:
            continue
          composition_candidates[label].append(crule)

      # we have a set of possible substitutions (of varying depth) for each
      # nonterminal; now we consider every possible way of combining them (the
      # Cartesian product of all the candidate lists)
      comp_cand_list = []
      label_list = []
      for label, comp_cand in composition_candidates.items():
        label_list.append(label)
        comp_cand_list.append(comp_cand)
      compositions = itertools.product(*comp_cand_list)
      compositions = list(compositions)

      # now actually create the composed rules
      for composition in compositions:
        cdict = dict(zip(label_list, composition))
        composed_rule = make_composed_rule(rule, cdict)
        composed_rules_this_depth.append(composed_rule)

    composed_rules += composed_rules_this_depth

  return [rule.canonicalize_amr() for rule in composed_rules]

def extract_rules(amr, tree, align, composition_depth):
  """
  Extracts all possible rules from the given tree-string pair.
  """
  rules = []
  frontier = frontier_edges(amr, tree, align)
  next_index = 0
  while frontier:
    rule, amr, tree, align, next_index = make_rule(frontier[0], amr, tree,
        align, next_index)
    rules.append(rule)
    frontier = frontier_edges(amr, tree, align)

  composed_rules = make_composed_rules(rules, composition_depth)
  return composed_rules

def collect_counts(rules):
  """
  Collects counts of the number of times each rule is used in the training data
  for the "observed derivation" ML estimate of rule weights.
  """
  rule_mapper = {}
  rule_counter = {}
  rule_normalizer = ddict(lambda:0.0)
  for rule in rules:
    rule_key = '%s:::%s:::%s' % (rule.symbol, rule.amr, rule.parse)
    rule_key = re.sub(r'\s+', ' ', rule_key)
    rule_key = re.sub(r'\[\d+\]', '[D]', rule_key)
    if rule_key not in rule_mapper:
      rule_mapper[rule_key] = rule
      rule_counter[rule_key] = 1
    else:
      rule_counter[rule_key] += 1
    rule_normalizer[rule.symbol] += 1

  grammar = {}
  next_id = 0
  for key in rule_mapper:
    rule = rule_mapper[key]
    count = rule_counter[key]
    norm = rule_normalizer[rule.symbol]
    g_rule = Rule(next_id, rule.symbol, float(count)/norm, rule.amr, rule.parse)
    grammar[next_id] = g_rule
    next_id += 1

  return grammar
  

if __name__ == "__main__":
    extractor = ExtractorSynSem()
    extractor.main(sys.argv)
