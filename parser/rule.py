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
    """
    Loads a SHRG grammar (a of rules keyed by rule number) from the given
    prefix. First attempts to load a pickled representation of the grammar, and
    falls back on plaintext. See documentation for format details.
    """

    # try loading from pickle
    try:
      pickle_file = open('%s.pickle' % prefix)
      output = pickle.load(pickle_file)
      pickle_file.close()
      return output
    except IOError:
      log.warn('No pickled grammar---loading from string instead. ' + \
          'This could take a while.')


    # try loading from plain text
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
    """
    Writes a SHRG grammar to a file, in both text and pickled formats.
    """

    pickle_file = open('%s.pickle' % prefix, 'w')
    lhs_file = open('%s.lhs' % prefix, 'w')
    rhs_amr_file = open('%s.rhs-amr' % prefix, 'w')
    rhs_ptb_file = open('%s.rhs-ptb' % prefix, 'w')

    for rule in grammar.values():
      print >>lhs_file, '%d,%s,%f' % (rule.rule_id, rule.symbol, rule.weight)
      print >>rhs_amr_file, '%d,%s' % (rule.rule_id, 
          re.sub(r'\s+', ' ', str(rule.amr)))
          #' '.join(str(rule.amr).split('\n')))
      print >>rhs_ptb_file, '%d,%s' % (rule.rule_id, 
          ' '.join(str(rule.parse).split('\n')))

    pickle.dump(grammar, pickle_file)
    lhs_file.close()
    rhs_amr_file.close()
    rhs_ptb_file.close()
    pickle_file.close()

  @classmethod
  def normalize_weights(cls, grammar):
    """
    Reweights the given grammar _conditionally_, so that the weights of all
    rules with the same right hand side sum to 1.
    """
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

