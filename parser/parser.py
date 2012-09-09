from lib.exceptions import InvocationException, InputFormatException
from lib.amr.dag import Dag
from nltk import Tree
from pyparsing import ParseException
from lib import log
import time
from collections import defaultdict as ddict, deque

from item import CfgItem, HergItem, CfgHergItem
from rule import Rule

IFORMAT_STRING = 0
IFORMAT_TREE = 1
IFORMAT_GRAPH = 2

OFORMAT_BOLINAS = 0
OFORMAT_CARMEL = 1
OFORMAT_TIBURON = 2
OFORMAT_CDEC = 3

class Parser:

  def __init__(self):
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
    if len(args) == 4:
      grammar_path, i1_path, output_prefix, output_format = args
      i2_path = None
    elif len(args) == 5:
      grammar_path, i1_path, i2_path, output_prefix, output_format = args
    else:
      raise InvocationException()
    i1_format = get_format(i1_path)
    i2_format = get_format(i2_path)

    # RTG parsing isn't supported yet
    if i2_path:
      if i1_format == IFORMAT_STRING and i2_format == IFORMAT_GRAPH:
        charts = parse_corpus(grammar_path, i1_path, i2_path)
      else:
        raise InputFormatException("If biparsing, must give [string, graph]" + \
            " as input.")
    else:
      if i1_format == IFORMAT_STRING:
        charts = parse_corpus(grammar_path, string_input_path=i1_path)
      elif i1_format == IFORMAT_GRAPH:
        charts = parse_corpus(grammar_path, graph_input_path=i1_path)
      else:
        raise InputFormatException("Must give one of string or graph as input.")

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

def parse_corpus(grammar_path, string_input_path=None, graph_input_path=None):
  """
  Finds derivation forests for all the examples in the corpus specified by one
  or both of the input_paths.
  """
  parse_string = True if string_input_path else False
  parse_graph = True if graph_input_path else False
  assert parse_string or parse_graph
  modes = []
  if parse_string:
    modes.append('string')
  if parse_graph:
    modes.append('graph')
  log.info('Parsing %s.' % ' and '.join(modes))

  # for more efficient filtering of rules, precompute a fast lookup of terminals
  grammar = Rule.load_from_file(grammar_path)
  if parse_string and parse_graph:
    filter_cache = make_synch_filter_cache()
  elif parse_string:
    filter_cache = make_string_filter_cache()
  else:
    filter_cache = make_graph_filter_cache()

  # get all of the input data into machine-readable format
  strings = []
  graphs = []
  if parse_string:
    with open(string_input_path) as sf:
      for line in sf.readlines():
        strings.append(line.strip().split())
  if parse_graph:
    with open(graph_input_path) as gf:
      for line in gf.readlines():
        graphs.append(Dag.from_string(line))

  # parse!
  charts = []
  for i in range(max(len(strings), len(graphs))):
    string = strings[i] if parse_string else None
    graph = graphs[i] if parse_graph else None
    chart = parse(grammar, string, graph, filter_cache)
    charts.append(chart)

  return charts

def successful_parse(string, graph, item, string_size, graph_size):
  """
  Determines whether the given item represents a complete derivation of the
  object(s) being parsed.
  """
  # make sure the right start symbol is used
  if 'root_ROOT' not in item.rule.symbol:
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

def parse(grammar, string, graph, filter_cache):
  """
  Parses the given string and/or graph with the provided grammar.
  """

  # This is a long function, so let's start with a high-level overview. This is
  # a "deductive-proof-style" parser: We begin with one "axiomatic" chart item
  # for each rule, and combine these items with each other and with fragments of
  # the object(s) being parsed to deduce new items. We can think of these items
  # as defining a search space in which we need to find a path to the goal item.
  # The parser implemented here performs a BFS of this search space.

  # remember when we started
  start_time = time.clock()
  log.info('parse')

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
  word_terminal_lookup = ddict(set) # a mapping from words to string indices
  nonterminal_lookup = ddict(set) # a mapping from labels to graph edges
  reverse_lookup = ddict(set) # a mapping from outside symbols open items
  if string:
    word_terminal_lookup = ddict(set)
    for i in range(len(string)):
      word_terminal_lookup[string[i]].add(i)
  if graph:
    edge_terminal_lookup = ddict(set)
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
    #log.debug('handling', item)

    if item.closed:
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
          if (item, oitem) in attempted:
            # don't repeat combinations we've tried before
            continue
          attempted.add((item, oitem))
          if not item.can_complete(oitem):
            continue
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
          chart[nitem].add((item,))
          if nitem not in pending and nitem not in visited:
            queue.append(nitem)
            pending.add(nitem)

  if success:
    log.info('success!')
  etime = time.clock() - start_time
  log.chatter('done in ' + str(etime) + 's')

  # TODO return partial chart
  return chart

def make_synch_filter_cache():
  pass

def make_string_filter_cache():
  pass

def make_graph_filter_cache():
  pass
