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

from item import Item, BoundaryItem
from rule import Rule

ITEM_CLASS = BoundaryItem

class ParserTD:

  def __init__(self):
    pass

  @classmethod
  def help(cls):
    return 'Usage: bolinas parse-td <grammar> <input1>'

  def main(self, *args):
    if len(args) != 2:
      raise InvocationException()

    grammar_path, input_path = args
    if not check_format(input_path):
      raise InvocationException("Only graph parsing is supported.")

    grammar = Rule.load_from_file(grammar_path)
    parse_corpus(grammar, input_path)

def parse_corpus(grammar, input_path):
  # TODO add back in?
  #filter_cache = make_graph_filter_cache()

  graphs = []
  with open(input_path) as graph_file:
    for line in graph_file.readlines():
      graphs.append(Dag.from_string(line))

  start_time = time.clock()
  charts = []
  for i in range(len(graphs)):
    graph = graphs[i]
    raw_chart = parse(grammar, graph)

  etime = time.clock() - start_time
  log.info('Parsed %s sentences in %.2fs' % (len(graphs), etime))

def parse(grammar, graph):
  """
  Parses the given graph with the provided grammar.
  """

  # This function is very similar to its counterpart in the regular
  # (non-tree-decomposing) parser. Read the comments there to understand how it
  # works.

  start_time = time.clock()
  log.chatter('parse...')

  # ensure that the input graph has its shortest-path table precomputed
  graph.compute_fw_table()

  chart = ddict(set)
  # TODO prune
  pgrammar = grammar.values() 
  queue = deque()
  pending = set()
  attempted = set()
  visited = set()
  terminal_lookup = ddict(set)
  passive_item_lookup = ddict(set)
  tree_node_lookup = ddict(set)
  passive_item_rev_lookup = ddict(set)
  tree_node_rev_lookup = ddict(set)

  for edge in graph.triples():
    terminal_lookup[edge[1]].add(edge)

  for rule in pgrammar:
    for leaf in rule.tree_leaves:
      axiom = ITEM_CLASS(rule, leaf, graph)
      queue.append(axiom)
      pending.add(axiom)
      assert leaf not in rule.tree_to_edge

  success = False

  while queue:
    item = queue.popleft()
    pending.remove(item)
    visited.add(item)
    log.debug('handling', item)

    if item.target == Item.NONE:
      log.debug('  none')
      tree_node_lookup[item.self_key].add(item)
      for ritem in tree_node_rev_lookup[item.self_key]:
        if ritem not in pending:
          queue.append(ritem)
          pending.add(ritem)

    elif item.target == Item.ROOT:
      log.debug('  root')
      if is_goal(item, graph):
        chart['START'].add((item,))
        success = True

      passive_item_lookup[item.self_key].add(item)
      for ritem in passive_item_rev_lookup[item.self_key]:
        if ritem not in pending:
          queue.append(ritem)
          pending.add(ritem)

    elif item.target == Item.TERMINAL:
      log.debug('  terminal')
      new_items = [item.terminal(edge) for edge in terminal_lookup[item.next_key]]
      new_items = [i for i in new_items if i]
      for nitem in new_items:
        chart[nitem].add((item,))
        if nitem not in pending and nitem not in visited:
          log.debug('    new item!', nitem)
          queue.append(nitem)
          pending.add(nitem)

    else:
      if item.target == Item.BINARY:
        log.debug('  binary')
        rev_lookup = tree_node_rev_lookup
        lookup = tree_node_lookup
        action = ITEM_CLASS.binary
      elif item.target == Item.NONTERMINAL:
        log.debug('  nonterminal')
        rev_lookup = passive_item_rev_lookup
        lookup = passive_item_lookup
        action = ITEM_CLASS.nonterminal
      else:
        assert False

      rev_lookup[item.next_key].add(item)
      for oitem in lookup[item.next_key]:
        if (item, oitem) in attempted:
          continue
        attempted.add((item, oitem))
        log.debug('  try', oitem)
        nitem = action(item, oitem)
        if not nitem:
          continue
        log.debug('    new item!', nitem)
        chart[nitem].add((item, oitem))
        if nitem not in pending and nitem not in visited:
          queue.append(nitem)
          pending.add(nitem)

  if success:
    log.chatter('  success!')

  etime = time.clock() - start_time
  log.chatter('done in %.2fs' % etime)

def is_goal(item, amr):
  if item.rule.symbol[:9] != 'root_ROOT':
    return False
  return item.matches_whole_graph()

def check_format(path):
  if path == None:
    return None
  with open(path) as f:
    line = f.readline().strip()
    try:
      Dag.from_string(line)
      return True
    except ParseException:
      return False
