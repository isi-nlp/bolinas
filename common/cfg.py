import itertools
from common import log

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
        Return the k-best derivations from this chart. 
        """

        if item == "START":
            rprob = 0.0 if logprob else 1.0
        else: 
            rprob = item.rule.weight

        # If item is a leaf, just return it and its probability    
        if not item in self: 
            if item == "START":
                log.info("No derivations.")
                return []
            else:
                return [(rprob, item)]

        pool = []
        # Find the k-best options for this rule.
        # Compute k-best for each possible split and add to pool.
        for split in self[item]:
            nts, children = zip(*split.items())
            kbest_each_child = [self.kbest(child, k, logprob) for child in children]
           
            generator = itertools.product(*kbest_each_child)
            for i in range(k):
                try:
                    cprobs, trees = zip(*next(generator))
                    if logprob:
                        prob = sum(cprobs) + rprob
                    else: 
                        prob = rprob
                        for p in cprobs: 
                            prob = prob * p
                    new_tree = (item, dict(zip(nts, trees)))
                    if new_tree:
                        pool.append((prob, new_tree))
                except StopIteration,e:
                    break
        
        # Return k-best from pool
        return sorted(pool, reverse=True)[:k]


    def inside_score(self, item, logprob = False):
        """
        Here we compute the inside scores for each nonterminal and split.
        """

        new_chart = dict()

        if item == "START":
            rprob = 0.0 if logprob else 1.0
        else: 
            rprob = item.rule.weight
       
        if not item in self: 
            if item == "START":
                return []
            else: 
                return []        
