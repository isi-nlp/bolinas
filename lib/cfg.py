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
        
