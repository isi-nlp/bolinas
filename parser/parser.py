import itertools
import math
import re
import time
from collections import defaultdict as ddict, deque
from lib.tree import Tree
from common import log
from common.hgraph.hgraph import Hgraph
from common.cfg import Chart
from vo_item import CfgItem, HergItem, SynchronousItem 
from vo_rule import VoRule
import pprint

class Parser:
  ''' 
  A deductive style parser for hypergraphs and strings that matches parts
  of the input hypergraph according to an arbitrary visit order for edges.
  (or left-to-right for strings, in which case this is essentially
  a CKY parser).
  '''

  def __init__(self, grammar):
    self.grammar = grammar
    self.nodelabels = grammar.nodelabels 

  def parse_graphs(self, graph_iterator):
      """
      Parse all the graphs in graph_iterator.
      This is a generator.
      """
      for graph in graph_iterator: 
          raw_chart = self.parse(None, graph)
          # The raw chart contains parser operations, need to decode the parse forest from this 
          yield cky_chart(raw_chart)

  def parse_strings(self, string_iterator):
    """
    Parse all strings in the string iterator.
    This is a generator.
    """
    for string in string_iterator:
        raw_chart = self.parse(string, None)
        yield cky_chart(raw_chart)

  def parse_bitexts(self, pair_iterator):
      """
      Parse all pairs of input objects returned by the pair iterator. 
      This is a generator.
      """ 
      for line1, line2 in pair_iterator:
          if self.grammar.rhs1_type == "hypergraph":
              obj1 = Hgraph.from_string(line1)
          else: 
              obj1 = line1.strip().split()
          
          if self.grammar.rhs2_type == "hypergraph":
              obj2 = Hgraph.from_string(line2)
          else: 
              obj2 = line2.strip().split()

          raw_chart = self.parse_bitext(obj1, obj2)
          yield cky_chart(raw_chart)

  def parse_bitext(self, obj1, obj2):
      """
      Parse a single pair of objects (two strings, two graphs, or string/graph).
      """
      rhs1type, rhs2type = self.grammar.rhs1_type, self.grammar.rhs2_type
      assert rhs1type in ["string","hypergraph"] and rhs2type in ["string","hypergraph"]
    
      # Remember size of input objects and figure out Item subclass
      if rhs1type == "string":
          obj1size = len(obj1) 
      elif rhs1type == "hypergraph":   
          obj1size = len(obj1.triples())
      if rhs2type == "string":
          obj2size = len(obj2)
      elif rhs2type == "hypergraph":
          obj2size = len(obj2.triples())
      grammar = self.grammar
      start_time = time.clock()
      log.chatter('parse...')

      # initialize data structures and lookups
      # we use various tables to provide constant-time lookup of fragments available
      # for shifting, completion, etc.
      chart = ddict(set)
        
      #TODO: command line filter to switch rule filter on/off
      pgrammar = [grammar[r] for r in grammar.reachable_rules(obj1, obj2)] #grammar.values()
      queue = deque() # the items left to be visited
      pending = set() # a copy of queue with constant-time lookup
      attempted = set() # a cache of previously-attempted item combinations
      visited = set() # a cache of already-visited items    
      nonterminal_lookup = ddict(set) # a mapping from labels to graph edges
      reverse_lookup = ddict(set) # a mapping from outside symbols to open items

      # mapping from words to string indices for each string
      word_terminal_lookup1 = ddict(set) 
      word_terminal_lookup2 = ddict(set) 

      if rhs1type == "string":
        for i in range(len(obj1)):
          word_terminal_lookup1[obj1[i]].add(i)
      
      if rhs2type == "string":
        for i in range(len(obj2)):
          word_terminal_lookup2[obj2[i]].add(i)
        
      # mapping from edge labels to graph edges for each graph
      edge_terminal_lookup1 = ddict(set) 
      edge_terminal_lookup2 = ddict(set) 

      if rhs1type == "hypergraph":
        for edge in obj1.triples(nodelabels = self.nodelabels):
          edge_terminal_lookup1[edge[1]].add(edge)

      if rhs2type == "hypergraph":
        for edge in obj2.triples(nodelabels = self.nodelabels):
          edge_terminal_lookup2[edge[1]].add(edge)

      for rule in pgrammar:
        item1class = CfgItem if rhs1type == "string" else HergItem
        item2class = CfgItem if rhs2type == "string" else HergItem
        axiom = SynchronousItem(rule, item1class, item2class, nodelabels = self.nodelabels)
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
          if self.successful_biparse(obj1, obj2, item, obj1size, obj2size):
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
              log.debug("  oitem:", oitem)
              if (item, oitem) in attempted:
                # don't repeat combinations we've tried before
                continue
              attempted.add((item, oitem))
              if not item.can_complete(oitem):
                log.debug("    fail")
                continue
              log.debug("    ok")
              nitem = item.complete(oitem)
              chart[nitem].add((item, oitem))
              if nitem not in pending and nitem not in visited:
                queue.append(nitem)
                pending.add(nitem)

          else:
               # shift ; this depends on the configuration (string/graph -> string/graph)
              if not item.outside1_is_nonterminal and not item.item1.closed: 
                    if rhs1type == "string":
                        new_items = [item.shift_word1(item.outside_object1, index) for index in
                        word_terminal_lookup1[item.outside_object1] if
                        item.can_shift_word1(item.outside_object1, index)]
                    else:
                        assert rhs1type is "hypergraph"
                        new_items = [item.shift_edge1(edge) for edge in
                          edge_terminal_lookup1[item.outside_object1] if
                          item.can_shift_edge1(edge)]
              else:
                    assert not item.outside2_is_nonterminal # Otherwise shift would not be called
                    if rhs2type == "string":
                        new_items = [item.shift_word2(item.outside_object2, index) for index in
                            word_terminal_lookup2[item.outside_object2] if
                            item.can_shift_word2(item.outside_object2, index)]
                    else: 
                        assert rhs2type is "hypergraph"
                        new_items = [item.shift_edge2(edge) for edge in
                            edge_terminal_lookup2[item.outside_object2] if
                            item.can_shift_edge2(edge)]

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

  def parse(self, string, graph):
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
        graph_size = len(graph.triples(nodelabels = self.nodelabels))
      else:
        graph_size = -1

      # initialize data structures and lookups
      # we use various tables to provide constant-time lookup of fragments available
      # for shifting, completion, etc.
      chart = ddict(set)
      
      # TODO: Command line option to switch grammar filter on/off
      if string:
          pgrammar = [grammar[r] for r in grammar.reachable_rules(string, None)] #grammar.values()
      if graph:
          pgrammar = [grammar[r] for r in grammar.reachable_rules(graph, None)] #grammar.values()
      
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
        for edge in graph.triples(nodelabels = self.nodelabels):
          edge_terminal_lookup[edge[1]].add(edge)
      for rule in pgrammar:
        axiom = axiom_class(rule, nodelabels = self.nodelabels)
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
          if self.successful_parse(string, graph, item, string_size, graph_size):
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
              log.debug("  oitem:", oitem)
              if (item, oitem) in attempted:
                # don't repeat combinations we've tried before
                continue
              attempted.add((item, oitem))
              if not item.can_complete(oitem):
                log.debug("    fail")
                continue
              log.debug("    ok")
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

  def successful_parse(self, string, graph, item, string_size, graph_size):
      """
      Determines whether the given item represents a complete derivation of the
      object(s) being parsed.
      """
      # make sure the right start symbol is used
      if self.grammar.start_symbol != item.rule.symbol:
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

  def successful_biparse(self, obj1, obj2, item, obj1size, obj2size):
      """
      Determines whether the given item represents a complete derivation of the
      object(s) being parsed.
      """
      # make sure the right start symbol is used
      if self.grammar.start_symbol != item.rule.symbol: 
          return False
      
      # make sure the item spans the whole object
      if item.item1class is CfgItem:
            if item.item1.j - item.item1.i != obj1size:
                return False
      else: 
            if len(item.item1.shifted) != obj1size:
                return False
      
      if item.item2class is CfgItem:
            if item.item2.j - item.item2.i != obj2size:
                return False
      else: 
            if len(item.item2.shifted) != obj2size:
                return False
      return True
      
        
        
#def output_bolinas(charts, grammar, prefix):
#  """
#  Prints given in native bolinas format.
#  """
#  raise InvocationException("Output format 'bolinas' is unsupported")

#def output_carmel(charts, grammar, prefix):
#  """
#  Prints given charts in carmel format, suitable for use with forest-em.
#  Will produce two files: prefix.carmel.norm (the RHS normalizer groups) and
#  prefix.carmel.charts (the charts).
#  """
#
#  # we need an explicit id for the start rule
#  # forest-em irritatingly expects rules to be 1-indexed rather than 0-indexed,
#  # so we have to increase all rule ids by 1
#  start_rule_id = max(grammar.keys()) + 2
#
#  # create the set of all normalization groups, and write them
#  normgroups = ddict(set)
#  normgroups['START'].add(start_rule_id)
#  for rule_id in grammar:
#    rule = grammar[rule_id]
#    normgroups[rule.symbol].add(rule.rule_id + 1)
#  with open('%s.carmel.norm' % prefix, 'w') as ofile:
#    print >>ofile, '(',
#    for group in normgroups.values():
#      print >>ofile, '(%s)' % (' '.join([str(rule) for rule in group])),
#    print >>ofile, ')'
#
#  # unlike the other formats, all carmel charts go in one file
#  with open('%s.carmel.charts' % prefix, 'w') as ofile:
#    for chart in charts:
#      # chart items we've already seen, and the labels assigned to them
#      seen = dict()
#      # python scoping weirdness requires us to store this variable with an
#      # extra layer of reference so that it can be reassigned by the inner
#      # method
#      next_id = [1]
#
#      def format_inner(item):
#        if item in seen:
#          return '#d' % seen[item]
#        my_id = next_id[0]
#        next_id[0] += 1
#        if item == 'START':
#          sym = start_rule_id
#        else:
#          # see note above on rule ids
#          sym = item.rule.rule_id + 1
#        if item in chart:
#          parts = []
#          for production in chart[item]:
#            prod_parts = []
#            for pitem in production:
#              prod_parts.append(format_inner(pitem))
#            parts.append('(%s %s)' % (sym, ' '.join(prod_parts)))
#          if len(parts) > 1:
#            return '#%d(OR %s)' % (my_id, ' '.join(parts))
#          else:
#            return '#%d%s' % (my_id, parts[0])
#        else:
#          return '#%d(%s)' % (my_id, sym)
#
#      print >>ofile, format_inner('START')


#def output_tiburon(charts, grammar, prefix):
#  """
#  Prints given charts in tiburon format, for finding n-best AMRs.
#  """
#
#  def start_stringifier(rhs_item):
#    return 'START -> %s # 1.0' % rhs_item.uniq_str()
#
#  def nt_stringifier(item, rhs):
#    nrhs = ' '.join([i for i in item.rule.string if i[0] == '#'])
#    # strip indices
#    nrhs = re.sub(r'\[\d+\]', '', nrhs)
#    for ritem in rhs:
#      # replace only one occurrence, in case we have a repeated NT symbol
#      nrhs = re.sub('#' + ritem.rule.symbol, ritem.uniq_str(), nrhs, count=1)
#    nrhs = '%s(%d(%s))' % (item.rule.symbol, item.rule.rule_id, nrhs)
#    return '%s -> %s # %f' % (item.uniq_str(), nrhs, item.rule.weight)
#
#  def t_stringifier(item):
#    return '%s -> %s(%d) # %f' % (item.uniq_str(), item.rule.symbol,
#        item.rule.rule_id, item.rule.weight)
#
#
#  for i, chart in zip(range(len(charts)), charts):
#    if chart:   
#        with open('%s%d.tiburon' % (prefix, i), 'w') as ofile:
#          rules = ['START'] + strings_for_items(chart, start_stringifier,
#              nt_stringifier, t_stringifier)
#          print >>ofile, '\n'.join(rules)

#def output_cdec(charts, grammar, prefix):
#  """
#  Prints given charts in cdec format, for finding n-best strings.
#  """
#
#  def start_stringifier(rhs_item):
#    return '[START] ||| [%s] ||| Rule=0.0' % rhs_item.uniq_str()
#
#  def nt_stringifier(item, rhs):
#    nrhs = ' '.join(item.rule.string)
#    # strip indices
#    nrhs = re.sub(r'\[\d+\]', '', nrhs)
#    for ritem in rhs:
#      # replace only one occurrence, in case we have a repeated NT symbol
#      nrhs = re.sub('#' + ritem.rule.symbol, '[%s]' % ritem.uniq_str(), nrhs)
#    return '[%s] ||| %s ||| Rule=%f' % (item.uniq_str(), nrhs,
#        math.log(item.rule.weight))
#
#  def t_stringifier(item):
#    return '[%s] ||| %s ||| Rule=%f' % (item.uniq_str(),
#        ' '.join(item.rule.string), math.log(item.rule.weight))
#
#  for i, chart in zip(range(len(charts)), charts):
#    with open('%s%d.cdec' % (prefix, i), 'w') as ofile:
#      rules = ['[S] ||| [START]'] + strings_for_items(chart, start_stringifier,
#          nt_stringifier, t_stringifier)
#      print >>ofile, '\n'.join(rules)

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


def cky_chart(chart):
  """
  Convert the chart returned by the parser into a standard parse chart.
  """

  def search_productions(citem, chart):
      if len(chart[citem]) == 0:
           return [] 
      if citem == "START":
           return [{"START":child[0]} for child in chart[citem]]
      
      prodlist = list(chart[citem])

      lefts = set(x[0] for x in prodlist)
      lengths = set(len(x) for x in prodlist)
      assert len(lengths) == 1
      split_len = lengths.pop()
      
      # figure out all items that could have been used to complete this nonterminal 
      if split_len != 1:
        assert split_len == 2
        symbol = prodlist[0][0].outside_symbol, prodlist[0][0].outside_nt_index
        result = []
        for child in prodlist:
            other_nts = search_productions(child[0], chart)
            if other_nts: 
                for option in other_nts:
                   d = dict(option)
                   d[symbol] = child[1]
                   result.append(d)
            else:
                   result.append(dict([(symbol, child[1])]))
        return result
      else: 
        return search_productions(prodlist[0][0], chart)

  stack = ['START']
  visit_items = set()
  while stack:
    item = stack.pop()
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
    
    prods = search_productions(item, chart)
    if prods: 
        cky_chart[item] = prods

  return cky_chart
