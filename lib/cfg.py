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

