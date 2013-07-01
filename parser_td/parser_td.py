from lib.tree import Tree
from lib.pyparsing import ParseException
from lib import log
import time
from collections import defaultdict as ddict, deque
import itertools
import re
import math

from lib.cfg import Chart

from lib.exceptions import InvocationException, InputFormatException
from lib.hgraph.hgraph import Hgraph

from td_item import Item, BoundaryItem

ITEM_CLASS = Item #BoundaryItem

class ParserTD:
    """
    A hyperedge replacement grammar parser that matches a rule's right hand side
    in the order of its tree decomposition. The algorithm implemented here is 
    described in the ACL 2013 paper.
    """
    def __init__(self, grammar):
        self.grammar = grammar
        self.nodelabels = grammar.nodelabels

    def parse_graphs(self, graph_iterator):
        """
        Parse all the graphs in graph_iterator.
        This is a generator.
        """
        #filter_cache = make_graph_filter_cache() 
        for graph in graph_iterator: 
            raw_chart = self.parse(graph)
            # The raw chart contains parser operations, need to decode the parse forest from this 
            yield td_chart_to_cky_chart(raw_chart)

    def parse_strings(self, string_iterator):
        """
        Not supported.
        """
        raise NotImplementedError, "Can't parse strings with tree decomposition hypergraph parser."

    def parse(self, graph):
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
      pgrammar = self.grammar.values() 
      queue = deque()
      pending = set()
      attempted = set()
      visited = set()
      terminal_lookup = ddict(set)
      passive_item_lookup = ddict(set)
      tree_node_lookup = ddict(set)
      passive_item_rev_lookup = ddict(set)
      tree_node_rev_lookup = ddict(set)

      for edge in graph.triples(nodelabels = self.nodelabels):
        terminal_lookup[edge[1]].add(edge)

      for rule in pgrammar:
        for leaf in rule.tree_leaves:
          axiom = ITEM_CLASS(rule, leaf, graph, nodelabels = self.nodelabels)
          queue.append(axiom)
          pending.add(axiom)
          assert leaf not in rule.tree_to_edge

      success = False

      while queue:
        item = queue.popleft()
        pending.remove(item)
        visited.add(item)
        log.debug('handling', item, item.subgraph)

        if item.target == Item.NONE:
          log.debug('  none')
          tree_node_lookup[item.self_key].add(item)
          for ritem in tree_node_rev_lookup[item.self_key]:
            if ritem not in pending:
              queue.append(ritem)
              pending.add(ritem)

        elif item.target == Item.ROOT:
          log.debug('  root')
          if self.is_goal(item):
            chart['START'].add((item,))
            success = True
            log.debug("success!")

          passive_item_lookup[item.self_key].add(item)
          for ritem in passive_item_rev_lookup[item.self_key]:
            if ritem not in pending:
              log.debug('    retrieving', ritem)
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
            log.debug('  try', oitem, oitem.subgraph)
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
      return chart

    def is_goal(self, item):
       if self.grammar.start_symbol !=  item.rule.symbol: 
          return False
       
       return item.matches_whole_graph()

def td_chart_to_cky_chart(chart):
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
      if not (item == 'START' or item.target in [Item.ROOT]):
        continue
      # this dictionary will store the nonterminals replacements used to create this item
      real_productions = {} 

      # we will search down the list of completions
      pitem_history = set()
      todo = [item]

      while todo:
        pitem = todo.pop()  
        # if this item has no children, there's nothing left to do with the
        # production
        if len(chart[pitem]) == 0:
            continue
        elif pitem == 'START':
          # add all START -> (real start symbol) productions on their own
          real_productions['START'] = list(sum(chart[pitem],()))
          break
  
        #elif pitem.rule.symbol == 'PARTIAL':
        #  assert len(chart[pitem]) == 1
        #  prod = list(chart[pitem])[0]
        #  for p in prod:
        #    real_productions.append([p])
        #  break
  
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
  
        #if split_len != 1:
        prodlist = list(chart[pitem])
            
        if prodlist[0][0].target == Item.NONTERMINAL:
            assert split_len == 2
            symbol = prodlist[0][0].outside_symbol, prodlist[0][0].outside_index
            production = [x[1] for x in chart[pitem]]
            real_productions[symbol] = production
            assert prodlist[0][1].target == Item.ROOT
            # move down.
            todo.append(prodlist[0][0])
    
        elif prodlist[0][0].target == Item.BINARY:
            assert split_len == 2
            for x in prodlist: 
                assert len(x) == 2
                todo.append(x[0])
                todo.append(x[1])
                 
      if real_productions:
          cky_chart[item] = real_productions 
    return cky_chart


