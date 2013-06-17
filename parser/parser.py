from lib.tree import Tree
from lib.pyparsing import ParseException
from lib import log
import time
from collections import defaultdict as ddict, deque
import itertools
import re
import math

from lib.exceptions import InvocationException, InputFormatException
from lib.amr.dag import Dag

from item import CfgItem, HergItem, CfgHergItem, Chart
from rule import Rule

import pprint

# input corpus formats
IFORMAT_STRING = 0
IFORMAT_TREE = 1
IFORMAT_GRAPH = 2

# output chart formats
OFORMAT_BOLINAS = 0
OFORMAT_CARMEL = 1
OFORMAT_TIBURON = 2
OFORMAT_CDEC = 3

# mappings from format names to constants
OFORMATS = {
  'bolinas': OFORMAT_BOLINAS,
  'carmel': OFORMAT_CARMEL,
  'tiburon': OFORMAT_TIBURON,
  'cdec': OFORMAT_CDEC
}

# OUTPUT_METHODS (mappings from formats to printing functions) at bottom of file
def get_format(path):
  """
  Determines the format (string, tree or graph) of the file at path.
  """
  if path == None:
    return None
  with open(path) as f:
    line = f.readline().strip()
    try:
      Dag.from_string(line)
      return IFORMAT_GRAPH
    except ParseException:
      pass
    try:
      Tree(line)
      return IFORMAT_TREE
    except ValueError as e:
      pass
    return IFORMAT_STRING


#def parse_corpus(string_input_path=None, graph_input_path=None):
#  """
#  Finds derivation forests for all the examples in the corpus specified by one
#  or both of the input_paths.
#  """
#  parse_string = True if string_input_path else False
#  parse_graph = True if graph_input_path else False
#  assert parse_string or parse_graph
#  modes = []
#  if parse_string:
#    modes.append('string')
#  if parse_graph:
#    modes.append('graph')
#  log.info('Parsing %s.' % ' and '.join(modes))
#
#  # for more efficient filtering of rules, precompute a fast lookup of terminals
#  if parse_string and parse_graph:
#    filter_cache = make_synch_filter_cache()
#  elif parse_string:
#    filter_cache = make_string_filter_cache()
#  else:
#    filter_cache = make_graph_filter_cache()
#
#  # get all of the input data into machine-readable format
#  strings = []
#  graphs = []
#  if parse_string:
#    with open(string_input_path) as sf:
#      for line in sf.readlines():
#        strings.append(line.strip().split())
#  if parse_graph:
#    with open(graph_input_path) as gf:
#      for line in gf.readlines():
#        graphs.append(Dag.from_string(line))
#
#  # parse!
#  start_time = time.clock()
#  charts = []
#  for i in range(max(len(strings), len(graphs))):
#    string = strings[i] if parse_string else None
#    graph = graphs[i] if parse_graph else None
#    raw_chart = parse(grammar, string, graph, filter_cache)
#    chart = cky_chart(raw_chart)
#    charts.append(chart)
#
#  etime = time.clock() - start_time
#  log.info('Parsed %s sentences in %.2fs' % (len(graphs), etime))
#
#  return charts

class Parser:

  def __init__(self, grammar):
    self.grammar = grammar
    pass

  @classmethod 
  def help(cls):
    return 'Usage: bolinas parse <grammar> <input1> [input2] ' + \
        '<output_prefix> <format>'

  def main(self, *args):
    """
    Parses (or biparses) the given input, and writes the resulting charts into
    files specified by the given output prefix and format.
    """
    # determine input format
    if len(args) == 4:
      grammar_path, i1_path, output_prefix, output_format = args
      i2_path = None
    elif len(args) == 5:
      grammar_path, i1_path, i2_path, output_prefix, output_format = args
    else:
      raise InvocationException()
    i1_format = get_format(i1_path)
    i2_format = get_format(i2_path)

    # determine output format
    if output_format not in OFORMATS:
      raise InputFormatException("output_format must be one of carmel, " + \
          "tiburon, bolinas or cdec")
    o_format = OFORMATS[output_format]

    grammar = Rule.load_from_file(grammar_path)

    # parse corpus
    # RTG parsing isn't supported yet
    if i2_path:
      if i1_format == IFORMAT_STRING and i2_format == IFORMAT_GRAPH:
        charts = parse_corpus(grammar, i1_path, i2_path)
      else:
        raise InputFormatException("If biparsing, must give [string, graph]" + \
            " as input.")
    else:
      if i1_format == IFORMAT_STRING:
        charts = parse_corpus(grammar, string_input_path=i1_path)
      elif i1_format == IFORMAT_GRAPH:
        charts = parse_corpus(grammar, graph_input_path=i1_path)
      else:
        raise InputFormatException("Must give one of string or graph " + \
            "as input.")

    # write output
    OUTPUT_METHODS[o_format](charts, grammar, output_prefix)

  def parse_graphs(self, graph_iterator):
      """
      Parse all the graphs in graph_iterator.
      This is a generator.
      """
      filter_cache = make_graph_filter_cache()
      for graph in graph_iterator: 
          raw_chart = self.parse(None, graph, filter_cache)
          # The raw chart contains parser operations, need to decode the parse forest from this 
          chart = cky_chart(raw_chart)
          yield chart

  def parse(self, string, graph, filter_cache):
      """
      Parses the given string and/or graph.
      """

      # This is a long function, so let's start with a high-level overview. This is
      # a "deductive-proof-style" parser: We begin with one "axiomatic" chart item
      # for each rule, and combine these items with each other and with fragments of
      # the object(s) being parsed to deduce new items. We can think of these items
      # as defining a search space in which we need to find a path to the goal item.
      # The parser implemented here performs a BFS of this search space.

      grammar = self.grammar

      # remember when we started
      start_time = time.clock()
      log.chatter('parse...')

      # specify what kind of items we're working with
      if string and graph:
        axiom_class = CfgHergItem
      elif string:
        axiom_class = CfgItem
      else:
        axiom_class = HergItem

      # remember the size of the example
      if string:
        string_size = len(string)
      else:
        string_size = -1
      if graph:
        graph_size = len(graph.triples())
      else:
        graph_size = -1

      # initialize data structures and lookups
      # we use various tables to provide constant-time lookup of fragments available
      # for shifting, completion, etc.
      chart = ddict(set)
      # TODO prune
      pgrammar = grammar.values()
      queue = deque() # the items left to be visited
      pending = set() # a copy of queue with constant-time lookup
      attempted = set() # a cache of previously-attempted item combinations
      visited = set() # a cache of already-visited items
      word_terminal_lookup = ddict(set) 
      nonterminal_lookup = ddict(set) # a mapping from labels to graph edges
      reverse_lookup = ddict(set) # a mapping from outside symbols open items
      if string:
        word_terminal_lookup = ddict(set) # mapping from words to string indices
        for i in range(len(string)):
          word_terminal_lookup[string[i]].add(i)
      if graph:
        edge_terminal_lookup = ddict(set) # mapping from edge labels to graph edges
        for edge in graph.triples():
          edge_terminal_lookup[edge[1]].add(edge)
      for rule in pgrammar:
        axiom = axiom_class(rule)
        queue.append(axiom)
        pending.add(axiom)
        if axiom.outside_is_nonterminal:
          reverse_lookup[axiom.outside_symbol].add(axiom)

      # keep track of whether we found any complete derivation
      success = False

      # parse
      while queue:
        item = queue.popleft()
        pending.remove(item)
        visited.add(item)
        log.debug('handling', item)

        if item.closed:
          log.debug('  is closed.')
          # check if it's a complete derivation
          if successful_parse(string, graph, item, string_size, graph_size):
              chart['START'].add((item,))
              success = True

          # add to nonterminal lookup
          nonterminal_lookup[item.rule.symbol].add(item)

          # wake up any containing rules
          # Unlike in ordinary state-space search, it's possible that we will have
          # to re-visit items which couldn't be merged with anything the first time
          # we saw them, and are waiting for the current item. The reverse_lookup
          # indexes all items by their outside symbol, so we re-append to the queue
          # all items looking for something with the current item's symbol.
          for ritem in reverse_lookup[item.rule.symbol]:
            if ritem not in pending:
              queue.append(ritem)
              pending.add(ritem)

        else:
          if item.outside_is_nonterminal:
            # complete
            reverse_lookup[item.outside_symbol].add(item)

            for oitem in nonterminal_lookup[item.outside_symbol]:
              log.debug("oitem:", oitem)
              if (item, oitem) in attempted:
                # don't repeat combinations we've tried before
                continue
              attempted.add((item, oitem))
              if not item.can_complete(oitem):
                log.debug("fail")
                continue
              log.debug("ok")
              nitem = item.complete(oitem)
              chart[nitem].add((item, oitem))
              if nitem not in pending and nitem not in visited:
                queue.append(nitem)
                pending.add(nitem)

          else:
            # shift
            if string and graph:
              if not item.outside_word_is_nonterminal:
                new_items = [item.shift_word(item.outside_word, index) for index in
                    word_terminal_lookup[item.outside_word] if
                    item.can_shift_word(item.outside_word, index)]
              else:
                assert not item.outside_edge_is_nonterminal
                new_items = [item.shift_edge(edge) for edge in
                    edge_terminal_lookup[item.outside_edge] if
                    item.can_shift_edge(edge)]               
            elif string:
              new_items = [item.shift(item.outside_word, index) for index in
                  word_terminal_lookup[item.outside_word] if
                  item.can_shift(item.outside_word, index)]
            else:
              assert graph
              new_items = [item.shift(edge) for edge in
                  edge_terminal_lookup[item.outside_edge] if
                  item.can_shift(edge)]

            for nitem in new_items:
              log.debug('  shift', nitem, nitem.shifted)
              chart[nitem].add((item,))
              if nitem not in pending and nitem not in visited:
                queue.append(nitem)
                pending.add(nitem)

      if success:
        log.chatter('  success!')
      etime = time.clock() - start_time
      log.chatter('done in %.2fs' % etime)

      # TODO return partial chart
      return chart



def successful_parse(string, graph, item, string_size, graph_size):
  """
  Determines whether the given item represents a complete derivation of the
  object(s) being parsed.
  """
  # make sure the right start symbol is used
  if 's' not in item.rule.symbol.lower():
    return False
    

  # make sure the item spans the whole object
  if string and graph:
    whole_string = item.cfg_item.j - item.cfg_item.i == string_size
    whole_graph = len(item.herg_item.shifted) == graph_size
    return whole_string and whole_graph
  elif string:
    return item.j - item.i == string_size
  else: # graph
    return len(item.shifted) == graph_size

def make_synch_filter_cache():
  pass

def make_string_filter_cache():
  pass

def make_graph_filter_cache():
  pass

def cky_chart(chart):
  stack = ['START']
  visit_items = set()
  while stack:
    item  = stack.pop()
    if item in visit_items:
      continue
    visit_items.add(item)
    for production in chart[item]:
      for citem in production:
        stack.append(citem)

  cky_chart = Chart() 
  for item in visit_items:
    # we only care about complete steps, so only add closed items to the chart
    if not (item == 'START' or item.closed):
      continue
    # this list will store the complete steps used to create this item
    real_productions = []
    # we will search down the list of completions
    pitem_history = set()
    pitem = item
    while True:

      # if this item has no children, there's nothing left to do with the
      # production
      if len(chart[pitem]) == 0:
        break
      elif pitem == 'START':
        # add all START -> (real start symbol) productions on their own
        real_productions.append(list(sum(chart[pitem],())))# ,()))
        break

      elif pitem.rule.symbol == 'PARTIAL':
        assert len(chart[pitem]) == 1
        prod = list(chart[pitem])[0]
        for p in prod:
          real_productions.append([p])
        break

      # sanity check: is the chain of derivations for this item shaped the way
      # we expect?
      lefts = set(x[0] for x in chart[pitem])
      lengths = set(len(x) for x in chart[pitem])
      # TODO might merge from identical rules grabbing different graph
      # components. Do we lose information by only taking the first
      # (lefts.pop(), below)?
      # TODO when this is fixed, add failure check back into topo_sort
      #assert len(lefts) == 1
      assert len(lengths) == 1
      split_len = lengths.pop()

      # figure out all items that could have been used to complete this rule
      if split_len != 1:
        assert split_len == 2
        production = [x[1] for x in chart[pitem]]
        real_productions.append(production)

      # move down the chain
      pitem = lefts.pop()

    # realize all possible splits represented by this chart item
    #all_productions = list(itertools.product(*real_productions))
    #if all_productions != [()]:
    #  cky_chart[item] = all_productions
    if real_productions:
        cky_chart[item] = real_productions 

  return cky_chart

def output_bolinas(charts, grammar, prefix):
  """
  Prints given in native bolinas format.
  """
  raise InvocationException("Output format 'bolinas' is unsupported")

def output_carmel(charts, grammar, prefix):
  """
  Prints given charts in carmel format, suitable for use with forest-em.
  Will produce two files: prefix.carmel.norm (the RHS normalizer groups) and
  prefix.carmel.charts (the charts).
  """

  # we need an explicit id for the start rule
  # forest-em irritatingly expects rules to be 1-indexed rather than 0-indexed,
  # so we have to increase all rule ids by 1
  start_rule_id = max(grammar.keys()) + 2

  # create the set of all normalization groups, and write them
  normgroups = ddict(set)
  normgroups['START'].add(start_rule_id)
  for rule_id in grammar:
    rule = grammar[rule_id]
    normgroups[rule.symbol].add(rule.rule_id + 1)
  with open('%s.carmel.norm' % prefix, 'w') as ofile:
    print >>ofile, '(',
    for group in normgroups.values():
      print >>ofile, '(%s)' % (' '.join([str(rule) for rule in group])),
    print >>ofile, ')'

  # unlike the other formats, all carmel charts go in one file
  with open('%s.carmel.charts' % prefix, 'w') as ofile:
    for chart in charts:
      # chart items we've already seen, and the labels assigned to them
      seen = dict()
      # python scoping weirdness requires us to store this variable with an
      # extra layer of reference so that it can be reassigned by the inner
      # method
      next_id = [1]

      def format_inner(item):
        if item in seen:
          return '#d' % seen[item]
        my_id = next_id[0]
        next_id[0] += 1
        if item == 'START':
          sym = start_rule_id
        else:
          # see note above on rule ids
          sym = item.rule.rule_id + 1
        if item in chart:
          parts = []
          for production in chart[item]:
            prod_parts = []
            for pitem in production:
              prod_parts.append(format_inner(pitem))
            parts.append('(%s %s)' % (sym, ' '.join(prod_parts)))
          if len(parts) > 1:
            return '#%d(OR %s)' % (my_id, ' '.join(parts))
          else:
            return '#%d%s' % (my_id, parts[0])
        else:
          return '#%d(%s)' % (my_id, sym)

      print >>ofile, format_inner('START')


def chart_to_tiburon(chart):
  def start_stringifier(rhs_item):
    return 'START -> %s # 1.0' % rhs_item.uniq_str()

  def nt_stringifier(item, rhs):
    nrhs = ' '.join([i for i in item.rule.string if i[0] == '#'])
    # strip indices
    nrhs = re.sub(r'\[\d+\]', '', nrhs)
    for ritem in rhs:
      # replace only one occurrence, in case we have a repeated NT symbol
      nrhs = re.sub('#' + ritem.rule.symbol, ritem.uniq_str(), nrhs, count=1)
    nrhs = '%s(%d(%s))' % (item.rule.symbol, item.rule.rule_id, nrhs)
    return '%s -> %s # %f' % (item.uniq_str(), nrhs, item.rule.weight)

  def t_stringifier(item):
    return '%s -> %s(%d) # %f' % (item.uniq_str(), item.rule.symbol,
        item.rule.rule_id, item.rule.weight)
  
  rules = ['START'] + strings_for_items(chart, start_stringifier,
            nt_stringifier, t_stringifier)
  return rules
    

def output_tiburon(charts, grammar, prefix):
  """
  Prints given charts in tiburon format, for finding n-best AMRs.
  """

  def start_stringifier(rhs_item):
    return 'START -> %s # 1.0' % rhs_item.uniq_str()

  def nt_stringifier(item, rhs):
    nrhs = ' '.join([i for i in item.rule.string if i[0] == '#'])
    # strip indices
    nrhs = re.sub(r'\[\d+\]', '', nrhs)
    for ritem in rhs:
      # replace only one occurrence, in case we have a repeated NT symbol
      nrhs = re.sub('#' + ritem.rule.symbol, ritem.uniq_str(), nrhs, count=1)
    nrhs = '%s(%d(%s))' % (item.rule.symbol, item.rule.rule_id, nrhs)
    return '%s -> %s # %f' % (item.uniq_str(), nrhs, item.rule.weight)

  def t_stringifier(item):
    return '%s -> %s(%d) # %f' % (item.uniq_str(), item.rule.symbol,
        item.rule.rule_id, item.rule.weight)


  for i, chart in zip(range(len(charts)), charts):
    if chart:   
        with open('%s%d.tiburon' % (prefix, i), 'w') as ofile:
          rules = ['START'] + strings_for_items(chart, start_stringifier,
              nt_stringifier, t_stringifier)
          print >>ofile, '\n'.join(rules)

def output_cdec(charts, grammar, prefix):
  """
  Prints given charts in cdec format, for finding n-best strings.
  """

  def start_stringifier(rhs_item):
    return '[START] ||| [%s] ||| Rule=0.0' % rhs_item.uniq_str()

  def nt_stringifier(item, rhs):
    nrhs = ' '.join(item.rule.string)
    # strip indices
    nrhs = re.sub(r'\[\d+\]', '', nrhs)
    for ritem in rhs:
      # replace only one occurrence, in case we have a repeated NT symbol
      nrhs = re.sub('#' + ritem.rule.symbol, '[%s]' % ritem.uniq_str(), nrhs)
    return '[%s] ||| %s ||| Rule=%f' % (item.uniq_str(), nrhs,
        math.log(item.rule.weight))

  def t_stringifier(item):
    return '[%s] ||| %s ||| Rule=%f' % (item.uniq_str(),
        ' '.join(item.rule.string), math.log(item.rule.weight))

  for i, chart in zip(range(len(charts)), charts):
    with open('%s%d.cdec' % (prefix, i), 'w') as ofile:
      rules = ['[S] ||| [START]'] + strings_for_items(chart, start_stringifier,
          nt_stringifier, t_stringifier)
      print >>ofile, '\n'.join(rules)

def strings_for_items(chart, start_stringifier, nt_stringifier, t_stringifier):
  strings = []
  stack = ['START']
  visited = set()
  while stack:
    item = stack.pop()
    if item in visited:
      continue
    visited.add(item)
    if item in chart:
      for rhs in chart[item]:
        if item == 'START':
          assert len(rhs) == 1
          strings.append(start_stringifier(rhs[0]))
          stack.append(rhs[0])
        else:
          strings.append(nt_stringifier(item, rhs))
          for ritem in rhs:
            assert ritem.rule.is_terminal or ritem in chart
            stack.append(ritem)
    else:
      assert item.rule.is_terminal
      strings.append(t_stringifier(item))

  return strings

# here so the relevant methods are in namespace when we create this dictionary
OUTPUT_METHODS = {
  OFORMAT_BOLINAS: output_bolinas,
  OFORMAT_CARMEL: output_carmel,
  OFORMAT_TIBURON: output_tiburon,
  OFORMAT_CDEC: output_cdec
}
