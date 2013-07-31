from common.hgraph.hgraph import NonterminalLabel
from common import log
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

class Item(object):
    pass

class HergItem(Item):
  """
  Chart item for a HRG parse.
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

    self.rev_mapping = dict((val, key) for key, val in mapping.items())

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
        other.shifted == self.shifted and \
        other.mapping == self.mapping

  def uniq_str(self):
    """
    Produces a unique string representation of this item. When representing
    charts in other formats (e.g. when writing a tiburon RTG file) we have to
    represent this item as a string, which we build from the rule id and list of
    nodes.
    """
    return 'R%d__%s' % (self.rule.rule_id, self.uniq_cover_str())


  def uniq_cover_str(self):
    edges = set()
    for head, elabel, tail in self.shifted:
        if tail:
            edges.add('%s:%s' % (head[0], ':'.join([x[0] for x in tail])))
        else: 
            edges.add(head[0])
    return ','.join(sorted(list(edges))) 

  def __repr__(self):
    return 'HergItem(%d, %d, %s, %s)' % (self.rule.rule_id, self.size, self.rule.symbol, len(self.shifted))

  def __str__(self):
    return '[%d, %d/%d, %s, {%s}]' % (self.rule.rule_id,
                                  self.size,
                                  len(self.rule.rhs1.triples()),
                                  self.outside_symbol,
                                  str([x for x in self.shifted]))

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
        if self.outside_triple[2]:
            o2, o2_labels = zip(*self.outside_triple[2])
        else: o2, o2_labels = [],[]
        if new_edge[2]:
            n2, n2_labels = zip(*new_edge[2])
        else: n2, n2_labels = [],[]
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

    #Make sure items are disjoint
    if any(edge in self.shifted for edge in new_item.shifted):
      #log.debug('fail bc overlap')
      return False

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

    #Check root label
    if self.nodelabels and o1label != new_item.rule.rhs1.node_to_concepts[nroot]: 
            return False

    if o1 in self.mapping and self.mapping[o1] != \
        new_item.mapping[list(new_item.rule.rhs1.roots)[0]]:
      #log.debug('fail bc mismapping')
      return False

    real_nroot = new_item.mapping[nroot]
    real_ntail = None
    for i in range(len(o2)):
      otail = o2[i]
      ntail = new_item.rule.rhs1.rev_external_nodes[i]
      #Check tail label
      if self.nodelabels and o2labels[i] != new_item.rule.rhs1.node_to_concepts[ntail]: 
        return False
      if otail in self.mapping and self.mapping[otail] != new_item.mapping[ntail]:
        #log.debug('fail bc bad mapping in tail')
        return False
   
    for node in new_item.mapping.values():
        if node in self.rev_mapping:
            onode =  self.rev_mapping[node]
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


class CfgItem(Item):
  """
  Chart item for a CFG parse.
  """

  def __init__(self, rule, size=None, i=None, j=None, nodelabels = False):
    # until this item is associated with some span in the sentence, let i and j
    # (the left and right boundaries) be -1
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

    self.shifted = []
    assert len(rule.rhs1) != 0

    if size == 0:
      assert i == -1
      assert j == -1
      self.closed = False
      self.outside_word = rule.rhs1[rule.rhs1_visit_order[0]]
    elif size < len(rule.string):
      self.closed = False
      self.outside_word = rule.string[rule.rhs1_visit_order[self.size]]
    else:
      self.closed = True
      self.outside_word = None

    if self.outside_word and isinstance(self.outside_word, NonterminalLabel):
      self.outside_is_nonterminal = True
      self.outside_symbol = self.outside_word.label
      self.outside_nt_index = self.outside_word.index
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
    return 'CfgItem(%d, %d, %s, (%d, %d))' % (self.rule.rule_id, self.size, str(self.closed), self.i, self.j)

  def __str__(self):
    return '[%s, %d/%d, (%d,%d)]' % (self.rule,
                                     self.size,
                                     len(self.rule.rhs1),
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

    return self.i == -1 or new_item.i == self.j #or new_item.j == self.i

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



class SynchronousItem(Item):
  """
  Chart item for a synchronous CFG/HRG parse. (Just a wrapper for paired
  CfgItem / HergItem.)
  """

  def __init__(self, rule, item1class, item2class, item1 = None, item2 = None, nodelabels = False):

    self.shifted = ([],[]) 

    self.rule = rule
    self.nodelabels = nodelabels
    self.item1class = item1class
    self.item2class = item2class

    if item1: 
        self.item1 = item1 
    else:
        self.item1 = item1class(rule.project_left(), nodelabels = nodelabels) 
    if item2: 
        self.item2 = item2
    else: 
        self.item2 = item2class(rule.project_right(), nodelabels = nodelabels)

    if self.item1.closed and self.item2.closed: 
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

    if item1class is CfgItem:
        self.outside1_is_nonterminal = self.item1.outside_is_nonterminal
        self.outside_object1 = self.item1.outside_word
    else:
        self.outside1_is_nonterminal = self.item1.outside_is_nonterminal
        self.outside_object1= self.item1.outside_triple[1] if \
            self.item1.outside_triple else None

    if item2class is CfgItem:
        self.outside2_is_nonterminal = self.item2.outside_is_nonterminal
        self.outside_object2 = self.item2.outside_word
    else:
        self.outside2_is_nonterminal = self.item2.outside_is_nonterminal
        self.outside_object2 = self.item2.outside_triple[1] if \
            self.item2.outside_triple else None
    
    self.outside_is_nonterminal = self.outside1_is_nonterminal and \
        self.outside2_is_nonterminal

    if self.outside_is_nonterminal:
      assert self.outside_object1 == self.outside_object2
      self.outside_symbol = self.item1.outside_symbol
      self.outside_nt_index = self.item1.outside_nt_index

    self.__cached_hash = None

  def uniq_str(self):
    """
    Produces a unique string representation of this item (see note on uniq_str
    in HergItem above).
    """
    edges = set()

    if item1class is CfgItem: 
        item1cover = "%d,%d" % (self.item1.i, self.item1.j)
    elif item1class is HergItem:
        item1cover = item1.uniq_cover_str(self)
    if item2class is CfgItem: 
        item2cover = "%d,%d" % (self.item2.i, self.item2.j)
    elif item2class is HergItem:
        item2cover = item2.uniq_cover_str(self)
    return '%d__%s__%s' % (self.rule.rule_id, item1cover, item2cover) 

  def __hash__(self):
    if not self.__cached_hash:
      self.__cached_hash = 2 * hash(self.item1) + 7 * hash(self.item2)
    return self.__cached_hash

  def __eq__(self, other):
    return isinstance(other, SynchronousItem) and other.item1 == self.item1 \
        and other.item2 == self.item2

  def __repr__(self):
    return "(%s, %s, %s, %s)" % (self.item1.__repr__(), self.item2.__repr__(), str(self.item1.closed),str(self.item2.closed))

  def can_shift_word1(self, word, index):
    """
    Determines whether given word, index can be shifted onto the CFG item.
    """
    assert isinstance(self.item1, CfgItem)
    return self.item1.can_shift(word, index)
  
  def can_shift_word2(self, word, index):
    """
    Determines whether given word, index can be shifted onto the CFG item.
    """
    assert isinstance(self.item2, CfgItem)
    return self.item2.can_shift(word, index)

  def shift_word1(self, word, index):
    """
    Shifts onto the CFG item.
    """
    assert isinstance(self.item1, CfgItem)
    nitem = self.item1.shift(word, index)
    self.shifted = (self.item1.shifted, self.item2.shifted)
    return SynchronousItem(self.rule, self.item1class, self.item2class, nitem, self.item2, nodelabels = self.nodelabels)
  
  def shift_word2(self, word, index):
    """
    Shifts onto the CFG item.
    """
    assert isinstance(self.item2, CfgItem)
    nitem = self.item2.shift(word, index)
    self.shifted = (self.item1.shifted, self.item2.shifted)
    return SynchronousItem(self.rule, self.item1class, self.item2class, self.item1, nitem, nodelabels = self.nodelabels)

  def can_shift_edge1(self, edge):
    """
    Determines whether the given edge can be shifted onto the HERG item.
    """
    assert isinstance(self.item1, HergItem)
    self.shifted = (self.item1.shifted, self.item2.shifted)
    return self.item1.can_shift(edge)
  
  def can_shift_edge2(self, edge):
    """
    Determines whether the given edge can be shifted onto the HERG item.
    """
    assert isinstance(self.item2, HergItem)
    self.shifted = (self.item1.shifted, self.item2.shifted)
    return self.item2.can_shift(edge)

  def shift_edge1(self, edge):
    """
    Shifts onto the HERG item.
    """
    nitem = self.item1.shift(edge)
    return SynchronousItem(self.rule, self.item1class, self.item2class, nitem, self.item2, nodelabels = self.nodelabels)
  
  def shift_edge2(self, edge):
    """
    Shifts onto the HERG item.
    """
    nitem = self.item2.shift(edge)
    return SynchronousItem(self.rule, self.item1class, self.item2class, self.item1, nitem, nodelabels = self.nodelabels)

  def can_complete(self, new_item):
    """
    Determines whether given item can complete both sides. 
    """
    if not (self.item1.can_complete(new_item.item1) and
            self.item2.can_complete(new_item.item2)):
        return False
    return True

  def complete(self, new_item):
    """
    Performs the synchronous completion, and gives back a new item.
    """
    nitem1 = self.item1.complete(new_item.item1)
    nitem2 = self.item2.complete(new_item.item2)
    return SynchronousItem(self.rule, self.item1class, self.item2class, nitem1, nitem2, nodelabels = self.nodelabels)
