from lib.exceptions import InputFormatException, BinarizationException, GrammarError

class Grammar(dict):

    def __init__(self, nodelabels = False):
        self.nodelabels = nodelabels  
 
    @classmethod
    def load_from_file(cls, in_file, reverse = False, nodelabels = False):
        """
        Loads a SHRG grammar from the given file. 
        See documentation for format details.
        """    

        output = Grammar(nodelabels = nodelabels)

        rule_count = 1
        line_count = 0
        is_synchronous = False

        rhs1_type = None
        rhs2_type = None
        GRAPH_FORMAT = "hypergraph" 
        STRING_FORMAT = "string"
        TREE_FORMAT = "tree"

        buf = StringIO.StringIO() 

        for line in in_file: 
            line_count += 1
            l = line.strip()
            if l:
                if "#" in l: 
                    content, comment = l.split("#",1)
                else: 
                    content = l
                buf.write(content.strip())
                if ";" in content:
                    rulestring = buf.getvalue()
                    try:
                        content, weights = rulestring.split(";",1)
                        weight = 1.0 if not weights else float(weights)
                    except:
                        raise GrammarError, \
            "Line %i, Rule %i: Error near end of line." % (line_count, rule_count)
                    
                    lhs, rhsstring = content.split("->")
                    lhs = lhs.strip()
                    if "|" in rhsstring:
                        if not is_synchronous and rule_count > 1:
                            raise GrammarError,\
           "Line %i, Rule %i: All or none of the rules need to have two RHSs." % (line_count, rule_count)
                        is_synchronous = True
                        try:
                            rhs1,rhs2 = rhsstring.split("|")
                        except:
                            raise GrammarError,"Only up to two RHSs are allowed in grammar file."
                    else: 
                        if is_synchronous and rule_count > 0:
                            raise ParserError,\
            "Line %i, Rule %i: All or none of the rules need to have two RHSs." % (line_count, rule_count)
                        is_synchronous = False
                        rhs1 = rhsstring
                        rhs2 = None                               
                    
                    try:    # If the first graph in the file cannot be parsed, assume it's a string
                        r1  = Hgraph.from_string(rhs1)
                        r1_nts = set([(ntlabel.label, ntlabel.index) for h, ntlabel, t in r1.nonterminal_edges()])
                        if not rhs1_type:
                            rhs1_type = GRAPH_FORMAT
                    except ParserError, e: 
                        if rhs1_type == GRAPH_FORMAT:
                           raise ParserError,\
            "Line %i, Rule %i: Could not parse graph description: %s" % (line_count, rule_count, e.message)
                        else:
                           r1 = parse_string(rhs1) 
                           nts = [t for t in r1 if type(t) is NonterminalLabel]
                           r1_nts = set([(ntlabel.label, ntlabel.index) for ntlabel in nts])
                           rhs1_type = STRING_FORMAT
  
                    if is_synchronous: 
                        try:    # If the first graph in the file cannot be parsed, assume it's a string
                            r2  = Hgraph.from_string(rhs2)
                            r2_nts = set([(ntlabel.label, ntlabel.index) for h, ntlabel, t in r2.nonterminal_edges()])
                            if not rhs2_type:
                                rhs2_type = GRAPH_FORMAT
                        except ParserError, e: 
                            if rhs2_type == GRAPH_FORMAT:
                               raise ParserError,\
                "Line %i, Rule %i: Could not parse graph description: %s" % (line_count, rule_count, e.message)
                            else:
                               r2 = parse_string(rhs2) 
                               nts = [t for t in r2 if type(t) is NonterminalLabel]
                               r2_nts = set([(ntlabel.label, ntlabel.index) for ntlabel in nts])
                               rhs2_type = STRING_FORMAT

                        # Verify that nonterminals match up
                        if not r1_nts == r2_nts:
                            raise GrammarError, \
            "Line %i, Rule %i: Nonterminals do not match between RHSs." % (line_count, rule_count)
                    else: 
                        r2 = None
                    if is_synchronous and reverse: 
                        output[rule_count] = Rule(rule_count, lhs, weight, r2, r1, nodelabels = nodelabels) 
                    else: 
                        output[rule_count] = Rule(rule_count, lhs, weight, r1, r2, nodelabels = nodelabels) 
                    buf = StringIO.StringIO() 
                    rule_count += 1
        output.is_synchronous = is_synchronous
        if is_synchronous and reverse:
            output.rhs1_type, output.rhs2_type = rhs2_type, rhs1_type
        else: 
            output.rhs1_type, output.rhs2_type = rhs1_type, rhs2_type

        return output 

