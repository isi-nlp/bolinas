from lib.tree import Tree
from common import log
import time
from collections import defaultdict as ddict, deque
import itertools
import re
import math
from common.cfg import Chart
from common.exceptions import InvocationException, InputFormatException
from common.hgraph.hgraph import Hgraph
from td_item import Item, BoundaryItem, FasterCheckBoundaryItem


class ParserTD:
    """
    A hyperedge replacement grammar parser that matches a rule's right hand side
    in the order of its tree decomposition. The algorithm implemented here is 
    described in the ACL 2013 paper.
    """

    item_class = FasterCheckBoundaryItem

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
            res = td_chart_to_cky_chart(raw_chart)
            yield res
 
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
          axiom = self.item_class(rule, leaf, graph, nodelabels = self.nodelabels)
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
            action = self.item_class.binary
          elif item.target == Item.NONTERMINAL:
            log.debug('  nonterminal')
            rev_lookup = passive_item_rev_lookup
            lookup = passive_item_lookup
            action = self.item_class.nonterminal
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
    """
    Convert the parsing chart returned by the tree decomposition based parser 
    into a standard parse chart.
    """
    def search_productions(citem, chart):
        """
        Find all the complete steps that could have produces this chart item.
        Returns a list of split options, each encoded as a dictionary mapping
        nonterminals to items. 
        """
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
        result = []    
        if prodlist[0][0].target == Item.NONTERMINAL:
            assert split_len == 2
            symbol = prodlist[0][0].outside_symbol, prodlist[0][0].outside_index
            for child in prodlist: 
                assert child[1].target == Item.ROOT
                other_nts = search_productions(child[0], chart) 
                if other_nts:
                    for option in other_nts:
                        d = dict(option)
                        d[symbol] = child[1]
                        result.append(d)
                else:
                        result.append(dict([(symbol, child[1])]))
            return result
    
        elif prodlist[0][0].target == Item.BINARY:
            assert split_len == 2
            for child in prodlist: 
                assert len(child) == 2
                r1 = search_productions(child[0], chart)
                r2 = search_productions(child[1], chart)
                if r1 and r2:   #possibilities: all combinations of assignments to NTs in the subtree
                    other_iterator = itertools.product(r1,r2)
                    for p1, p2 in other_iterator:
                        nts = dict(p1)
                        nts.update(p2)
                        result.append(nts)
                else:  # Only one of the subtrees has nonterminals. 
                    result.extend(r1)
                    result.extend(r2)
            return result            

        elif prodlist[0][0].target == Item.TERMINAL:
            for child in prodlist: 
                assert len(child) == 1
                other_nts = search_productions(child[0], chart)
                if other_nts:
                    result.extend(other_nts)
            return result

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
        # we only care about nonterminal steps, so only add closed items to the chart
        if not (item == 'START' or item.target == Item.ROOT):
            continue
        # this dictionary will store the nonterminals replacements used to create this item
        prods = search_productions(item, chart) 
        if prods:
            cky_chart[item] = prods
    return cky_chart
