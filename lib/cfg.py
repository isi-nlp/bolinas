import itertools
from lib import log

class NonterminalLabel(object):
    """
    There can be multiple nonterminal edges with the same symbol. Wrap the 
    edge into an object so two edges do not compare equal.
    Nonterminal edges carry a nonterminal symbol and an index that identifies
    it uniquely in a rule.
    """

    #label_matcher = re.compile("(?P<label>.*?)(\[(?P<index>.*)\])?$")

    def __init__(self, label, index = None):            
        # Parsing and setting the sychronization index is now handled
        # by the grammar parser
        
        #if index is not None:   
        self.label = label
        self.index = index  
        #else: 
        #    match = NonterminalLabel.label_matcher.match(label)
        #    self.index = match.group("index")
        #    self.label = match.group("label")

    def __eq__(self, other):
        try: 
            return self.label == other.label and self.index == other.index
        except AttributeError:
            return False     
    
    def __repr__(self):
        return "NT(%s)" % str(self)

    def __str__(self):
        if self.index is not None:
            return "%s$%s" % (str(self.label), str(self.index))
        else: 
            return "%s$" % str(self.label)

    def __hash__(self):
        return 83 * hash(self.label) + 17 * hash(self.index)


    @classmethod
    def from_string(cls, s):
        label, index = s.split("$") 
        return NonterminalLabel(label, index or None)


class Chart(dict):
    """
    A CKY style parse chart that can return k-best derivations and can return inside and outside probabilities.
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

    @classmethod
    def from_raw(cls, chart):
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
        real_productions = {} 
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
            real_productions['START'] = list(sum(chart[pitem],()))
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
            prodlist = list(chart[pitem])
            symbol = prodlist[0][0].outside_symbol, prodlist[0][0].outside_nt_index
            production = [x[1] for x in chart[pitem]]
            real_productions[symbol] = production

          # move down the chain
          pitem = lefts.pop()

        # realize all possible splits represented by this chart item
        #all_productions = list(itertools.product(*real_productions))
        #if all_productions != [()]:
        #  cky_chart[item] = all_productions
        if real_productions:
            cky_chart[item] = real_productions 

      return cky_chart
