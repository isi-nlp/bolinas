from lib.hgraph.hgraph import NonterminalLabel
from lib import log
import itertools

# Some general advice for reading this file:
#
# Every rule specifies some fragment of the object (graph, string or both) that
# is being parsed, as well as a visit order on the individual elements of that
# fragment (tokens or edges respectively). The number of elements already
# visited is called the "size" of this item, and an item with nothing left to
# visit is "closed". The visit order specifies an implicit binarization of the
# rule in question, by allowing the item to consume only one other object (which
# we call the "outside" of the item) at any given time.
#
# In consuming this object, we either "shift" a terminal element or "complete" a
# nonterminal (actually a closed chart item). Each of these steps produces a new
# chart item.


class HergItem():
  """
  Chart item for a HERG parse.
  """

  def __init__(self, rule, size=None, shifted=None, mapping=None, nodelabels=False):
    # by default start empty, with no part of the graph consumed
    if size == None:
      size = 0
    if shifted == None:
      shifted = frozenset()
    if mapping == None:
      mapping = dict()

    self.rule = rule
    self.size = size
    self.shifted = shifted
    self.mapping = mapping

    self.nodelabels = nodelabels

    # Store the nonterminal symbol and index of the previous complete 
    # on this item so we can rebuild the derivation easily

    triples = rule.rhs1.triples(nodelabels = nodelabels)
    self.outside_symbol = None
    if size < len(triples):
      # this item is not closed
      self.outside_triple = triples[rule.rhs1_visit_order[size]]
      self.outside_edge = self.outside_triple[1]
      self.closed = False
      self.outside_is_nonterminal = isinstance(self.outside_triple[1],
          NonterminalLabel)
      if self.outside_is_nonterminal:
        # strip the index off of the nonterminal label
        #self.outside_symbol = str(self.outside_triple[1])
        #self.outside_symbol = self.outside_symbol[1:].split('[')[0]
        self.outside_symbol = self.outside_triple[1].label
        self.outside_nt_index = self.outside_triple[1].index
    else:
      # this item is closed
      self.outside_triple = None
      self.outside_edge = None
      self.closed = True
      self.outside_is_nonterminal = False

    self.__cached_hash = None

  def __hash__(self):
    # memoize the hash function
    if not self.__cached_hash:
      self.__cached_hash = 2 * hash(self.rule) + 3 * self.size + \
          5 * hash(self.shifted)
    return self.__cached_hash

  def __eq__(self, other):
    return isinstance(other, HergItem) and \
        other.rule == self.rule and \
        other.size == self.size and \
        other.shifted == self.shifted

  def uniq_str(self):
    """
    Produces a unique string representation of this item. When representing
    charts in other formats (e.g. when writing a tiburon RTG file) we have to
    represent this item as a string, which we build from the rule id and list of
    nodes.
    """
    edges = set()
    for head, role, tail in self.shifted:
      edges.add('%s:%s' % (head, ':'.join(tail)))
    return '%d__%s' % (self.rule.rule_id, ','.join(sorted(list(edges))))

  def __repr__(self):
    return 'HergItem(%d, %d, %s)' % (self.rule.rule_id, self.size, self.rule.symbol)

  def __str__(self):
    return '[%s, %d/%d, %s, {%d}]' % (self.rule,
                                  self.size,
                                  len(self.rule.rhs1.triples()),
                                  self.outside_symbol,
                                  len(self.shifted))

  def can_shift(self, new_edge):
    """
    Determines whether new_edge matches the outside of this item, and can be
    shifted.
    """
    # can't shift into a closed item
    if self.closed:
      return False
    # can't shift an edge that is already inside this item
    if new_edge in self.shifted:
      return False
    olabel = self.outside_triple[1]
    nlabel = new_edge[1]
    # make sure new_edge mathes the outside label
    if olabel != nlabel:
      return False
    # make sure new_edge preserves a consistent mapping between the nodes of the
    # graph and the nodes of the rule
    if self.nodelabels:
        o1, o1_label = self.outside_triple[0]
        n1, n1_label = new_edge[0]
        if o1_label != n1_label:
            return False
    else:
        o1 = self.outside_triple[0]
        n1 = new_edge[0]

    if o1 in self.mapping and self.mapping[o1] != n1:
      return False

    if self.nodelabels:
        o2, o2_labels = zip(*self.outside_triple[2])
        n2, n2_labels = zip(*new_edge[2])
        if o2_labels != n2_labels:
            return False 
    else:        
        o2 = self.outside_triple[2]
        n2 = new_edge[2]

        if len(o2) != len(n2):
            return False
            
        for i in range(len(o2)): 
            if o2[i] in self.mapping and self.mapping[o2[i]] != n2[i]:
                return False

    return True

  def shift(self, new_edge):
    """
    Creates the chart item resulting from a shift of new_edge. Assumes
    can_shift returned true.
    """
    olabel = self.outside_triple[1]
    o1 = self.outside_triple[0][0] if self.nodelabels else self.outside_triple[0]
    o2 = tuple(x[0] for x in self.outside_triple[2]) if self.nodelabels else self.outside_triple[2] 
    
    nlabel = new_edge[1]
    n1 = new_edge[0][0] if self.nodelabels else new_edge[0]
    n2 = tuple(x[0] for x in new_edge[2]) if self.nodelabels else new_edge[2] 

    assert len(o2) == len(n2) 
    new_size = self.size + 1
    new_shifted = frozenset(self.shifted | set([new_edge]))
    new_mapping = dict(self.mapping)
    new_mapping[o1] = n1
    for i in range(len(o2)):
        new_mapping[o2[i]] = n2[i]

    return HergItem(self.rule, new_size, new_shifted, new_mapping, self.nodelabels)

  def can_complete(self, new_item):
    """
    Determines whether new_item matches the outside of this item (i.e. if the
    nonterminals match and the node mappings agree).
    """
    # can't add to a closed item
    if self.closed:
      #log.debug('fail bc closed')
      return False
    # can't shift an incomplete item
    if not new_item.closed:
      #log.debug('fail bc other not closed')
      return False

    # make sure labels agree
    if not self.outside_is_nonterminal:
      #log.debug('fail bc outside terminal')
      return False

    #Make sure items are disjunct. THIS CAN BE SOLVED BY ALGORITHM 1 IN ACL PAPER.
    # TODO should be able to check boundary only 
    if any(edge in self.shifted for edge in new_item.shifted):
      #log.debug('fail bc overlap')
      return False

    rev_omap = dict([(val, key) for key, val in self.mapping.items()])
    
    # make sure mappings agree
    if self.nodelabels:
        o1, o1label = self.outside_triple[0]
        if self.outside_triple[2]:
            o2, o2labels = zip(*self.outside_triple[2])
        else: 
            o2, o2labels = [],[]
    else: 
        o1 = self.outside_triple[0]
        o2 = self.outside_triple[2]

    if len(o2) != len(new_item.rule.rhs1.external_nodes):
      #log.debug('fail bc hyperedge type mismatch')
      return False

    nroot = list(new_item.rule.rhs1.roots)[0]
    if self.nodelabels and o1label != new_item.rule.rhs1.node_to_concepts[nroot]:
            return False

    if o1 in self.mapping and self.mapping[o1] != \
        new_item.mapping[list(new_item.rule.rhs1.roots)[0]]:
      #log.debug('fail bc mismapping')
      return False

    real_nroot = new_item.mapping[nroot]
    # Prevent completions when the source node has been used already unless it's a reentrency
    #if real_nroot in rev_omap and rev_omap[real_nroot] != o1:
    #    return False

    real_ntail = None
    for i in range(len(o2)):
      otail = o2[i]
      ntail = new_item.rule.rhs1.rev_external_nodes[i]
      if self.nodelabels and o2labels[i] != new_item.rule.rhs1.node_to_concepts[ntail]:
        return False
      if otail in self.mapping and self.mapping[otail] != new_item.mapping[ntail]:
        #log.debug('fail bc bad mapping in tail')
        return False
      # Prevent completions when a tail node has been used already unless it's a reentrency
      #real_ntail = new_item.mapping[ntail]  
      #if real_ntail in rev_omap and rev_omap[real_ntail] != otail:
      #  return False
   
    for node in new_item.mapping.values():
        if node in rev_omap:
            onode =  rev_omap[node]
            if not (onode == o1 or onode in o2):
                return False

    return True

  def complete(self, new_item):
    """
    Creates the chart item resulting from a complete of new_item. Assumes
    can_shift returned true.
    """
    olabel = self.outside_triple[1]
    o1 = self.outside_triple[0][0] if self.nodelabels else self.outside_triple[0]
    o2 = tuple(x[0] for x in self.outside_triple[2]) if self.nodelabels else self.outside_triple[2] 
    
    new_size = self.size + 1
    new_shifted = frozenset(self.shifted | new_item.shifted)
    new_mapping = dict(self.mapping)
    new_mapping[o1] = new_item.mapping[list(new_item.rule.rhs1.roots)[0]]
    for i in range(len(o2)):
      otail = o2[i]
      ntail = new_item.rule.rhs1.rev_external_nodes[i]
      new_mapping[otail] = new_item.mapping[ntail]

    new = HergItem(self.rule, new_size, new_shifted, new_mapping, self.nodelabels)
    return new

class CfgItem():
  """
  Chart item for a CFG parse.
  """

  def __init__(self, rule, size=None, i=None, j=None):
    # until this item is associated with some span in the sentence, let i and j
    # (the left and right boundaries) be 0
    if size == None:
      size = 0
    if i == None:
      i = -1
    if j == None:
      j = -1

    self.rule = rule
    self.i = i
    self.j = j
    self.size = size

    assert len(rule.string) != 0

    if size == 0:
      assert i == -1
      assert j == -1
      self.closed = False
      self.outside_word = rule.string[rule.string_visit_order[0]]
    elif size < len(rule.string):
      self.closed = False
      self.outside_word = rule.string[rule.string_visit_order[self.size]]
    else:
      self.closed = True
      self.outside_word = None

    if self.outside_word and self.outside_word[0] == '#':
      self.outside_is_nonterminal = True
      # strip leading pound sign and NT index from label
      self.outside_symbol = self.outside_word[1:].split('[')[0]
    else:
      self.outside_is_nonterminal = False

    self.__cached_hash = None

  def __hash__(self):
    if not self.__cached_hash:
      self.__cached_hash = 2 * hash(self.rule) + 3 * self.i + 5 * self.j
    return self.__cached_hash

  def __eq__(self, other):
    return isinstance(other, CfgItem) and \
        other.rule == self.rule and \
        other.i == self.i and \
        other.j == self.j and \
        other.size == self.size

  def __repr__(self):
    return 'CfgItem(%d, %d)' % (self.rule.rule_id, self.size)

  def __str__(self):
    return '[%s, %d/%d, (%d,%d)]' % (self.rule,
                                     self.size,
                                     len(self.rule.string),
                                     self.i,self.j)

  def uniq_str(self):
    """
    Produces a unique string representation of this item (see note on uniq_str
    in HergItem above).
    """
    return '%d__%d_%d' % (self.rule.rule_id, self.i, self.j)

  def can_shift(self, word, index):
    """
    Determines whether word matches the outside of this item (i.e. is adjacent
    and has the right symbol) and can be shifted.
    """
    if self.closed:
      return False
    if self.i == -1:
      return True
    if index == self.i - 1:
      return self.outside_word == word
    elif index == self.j:
      return self.outside_word == word
    return False

  def shift(self, word, index):
    """
    Creates the chart item resulting from a shift of the word at the given
    index.
    """
    if self.i == -1:
      return CfgItem(self.rule, self.size+1, index, index+1)
    elif index == self.i - 1:
      return CfgItem(self.rule, self.size+1, self.i-1, self.j)
    elif index == self.j:
      return CfgItem(self.rule, self.size+1, self.i, self.j+1)
    assert False

  def can_complete(self, new_item):
    """
    Determines whether new_item matches the outside of this item.
    """
    if self.closed:
      return False
    if not new_item.closed:
      return False

    if self.outside_symbol != new_item.rule.symbol:
      return False

    return self.i == -1 or new_item.i == self.j or new_item.j == self.i

  def complete(self, new_item):
    """
    Creates the chart item resulting from a completion with the given item.
    """
    if self.i == -1:
      return CfgItem(self.rule, self.size+1, new_item.i, new_item.j)
    elif new_item.i == self.j:
      return CfgItem(self.rule, self.size+1, self.i, new_item.j)
    elif new_item.j == self.i:
      return CfgItem(self.rule, self.size+1, new_item.i, self.j)
    assert False


class CfgHergItem:
  """
  Chart item for a synchronous CFG/HERG parse. (Just a wrapper for paired
  CfgItem / HergItem.)
  """

  def __init__(self, rule, cfg_item=None, herg_item=None):
    if cfg_item == None:
      cfg_item = CfgItem(rule)
    if herg_item == None:
      herg_item = HergItem(rule)

    self.rule = rule
    self.cfg_item = cfg_item
    self.herg_item = herg_item

    if cfg_item.closed and herg_item.closed:
      self.closed = True
    else:
      self.closed = False

    # Now we potentially have two outsides---one in the graph and the other in
    # the string. The visit order will guarantee that if we first consume all
    # terminals in any order, the remainder of both string and graph visit
    # orders will agree on the sequence in which to consume nonterminals. (See
    # the Rule class.) Before consuming all terminals, it might be the case that
    # one item has a terminal outside and the other a nonterminal; in that case
    # we do not want an outside nonterminal associated with this item.

    self.outside_word_is_nonterminal = self.cfg_item.outside_is_nonterminal
    self.outside_edge_is_nonterminal = self.herg_item.outside_is_nonterminal
    self.outside_is_nonterminal = self.outside_word_is_nonterminal and \
        self.outside_edge_is_nonterminal

    self.outside_word = self.cfg_item.outside_word
    self.outside_edge = self.herg_item.outside_triple[1] if \
        self.herg_item.outside_triple else None

    if self.outside_is_nonterminal:
      assert cfg_item.outside_symbol == herg_item.outside_symbol
      self.outside_symbol = cfg_item.outside_symbol

    self.__cached_hash = None

  def uniq_str(self):
    """
    Produces a unique string representation of this item (see note on uniq_str
    in HergItem above).
    """
    edges = set()
    for head, role, tail in self.herg_item.shifted:
      edges.add('%s:%s' % (head, ':'.join(tail)))
    return '%d__%s__%d,%d' % (self.rule.rule_id, ','.join(sorted(list(edges))),
        self.cfg_item.i, self.cfg_item.j)

  def __hash__(self):
    if not self.__cached_hash:
      self.__cached_hash = 2 * hash(self.cfg_item) + 7 * hash(self.herg_item)
    return self.__cached_hash

  def __eq__(self, other):
    return isinstance(other, CfgHergItem) and other.cfg_item == self.cfg_item \
        and other.herg_item == self.herg_item

  def __repr__(self):
    if "outside_symbol" in self.__dict__:
        return '[%s, (%d,%d), (%s), {%d}]' % (self.rule, self.cfg_item.i, self.cfg_item.j,
            self.outside_symbol,
            len(self.herg_item.shifted))
    else: 
        return '[%s, (%d,%d), {%d}]' % (self.rule, self.cfg_item.i, self.cfg_item.j,
         len(self.herg_item.shifted))

  def can_shift_word(self, word, index):
    """
    Determines whether given word, index can be shifted onto the CFG item.
    """
    return self.cfg_item.can_shift(word, index)

  def shift_word(self, word, index):
    """
    Shifts onto the CFG item.
    """
    citem = self.cfg_item.shift(word, index)
    return CfgHergItem(self.rule, citem, self.herg_item)

  def can_shift_edge(self, edge):
    """
    Determines whether the given edge can be shifted onto the HERG item.
    """
    return self.herg_item.can_shift(edge)

  def shift_edge(self, edge):
    """
    Shifts onto the HERG item.
    """
    hitem = self.herg_item.shift(edge)
    return CfgHergItem(self.rule, self.cfg_item, hitem)

  def can_complete(self, new_item):
    """
    Determines whether given item can be shifted onto both the CFG item and the
    HERG item.
    """

    #print '  cfg:', self.cfg_item.can_complete(new_item.cfg_item)
    #print '  herg:', self.herg_item.can_complete(new_item.herg_item)
    if not (self.cfg_item.can_complete(new_item.cfg_item) and
            self.herg_item.can_complete(new_item.herg_item)):
      return False
    assert self.cfg_item.outside_word == \
        str(self.herg_item.outside_triple[1])
    return True

  def complete(self, new_item):
    """
    Performs the synchronous completion, and gives back a new item.
    """
    citem = self.cfg_item.complete(new_item.cfg_item)
    hitem = self.herg_item.complete(new_item.herg_item)
    return CfgHergItem(self.rule, citem, hitem)


class Chart(dict):
    """
    A CKY style parse chart
    """

    def kbest(self, item, k, logprob = False):
        """
        Find all 
        """

        if item == "START":
            rprob = 0.0 if logprob else 1.0
        else: 
            rprob = item.rule.weight

        # If item is a leaf, just return it and it's probability    
        if not item in self: 
            if item == "START":
                log.info("No derivations.")
                return []
            else:
                return [(rprob, item)]

        # Find the k-best options for each child
        nts = []
        kbest_each_child = []
        for nt in self[item]: 
            nts.append(nt)
            kbest_each_child.append(sorted(sum([self.kbest(poss,k, logprob) for poss in self[item][nt]],[]), reverse = True)[:k])

        # Compute cartesian product of possibilities among children. Evaluated lazily.
        generator = itertools.product(*kbest_each_child)

        # Select k-best and compute score
        kbest = [] 
        for i in range(k):
            try:
                cprobs, trees = zip(*next(generator)) #unpack list of (score, chart) tuples 
                if logprob:
                    prob = sum(cprobs) + rprob
                else:
                    prob = rprob
                    for p in cprobs: 
                        prob = prob * p 

                new_tree = (item, dict(zip(nts,trees)))
                if new_tree:
                    kbest.append((prob, new_tree))
            except StopIteration, e:
                break
        
        if item == "START" and len(kbest)<k:
                log.info("K-best did not find %i derivations. Returning best %i." % (k, len(kbest)))
        return kbest

#    def format_tiburon(chart):
#      def start_stringifier(rhs_item):
#        return 'START -> %s # 1.0' % rhs_item.uniq_str()
#
#      def nt_stringifier(item, rhs):
#       
#     
#        nrhs = '%s(%d(%s))' % (lhs, item.rule.rule_id, children)
#        return '%s -> %s # %f' % (item., nrhs, item.rule.weight)
#
#      def t_stringifier(item):
#        return '%s # %f' % (item.rule.rule_id, item.rule.weight)
#      
#      rules = ['START'] + chart.strings_for_items(start_stringifier,
#                nt_stringifier, t_stringifier)
#      return "\n".join(rules)
#        
#    def strings_for_items(chart, start_stringifier, nt_stringifier, t_stringifier):
#      strings = []
#      stack = ['START']
#      visited = set()
#      while stack:
#        item = stack.pop()
#        if item in visited:
#          continue
#        visited.add(item)
#        if item in chart:
#          for rhs in chart[item]:
#            if item == 'START':
#              assert len(rhs) == 1
#              strings.append(start_stringifier(rhs[0]))
#              stack.append(rhs[0])
#            else:
#              strings.append(nt_stringifier(item, rhs))
#              for ritem in rhs:
#                assert ritem.rule.is_terminal or ritem in chart
#                stack.append(ritem)
#        else:
#          assert item.rule.is_terminal
#          strings.append(t_stringifier(item))
#
#      return strings
