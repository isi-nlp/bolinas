from common.exceptions import InputFormatException, BinarizationException, GrammarError, ParserError, DerivationException
from common.hgraph.hgraph import Hgraph
from common.cfg import NonterminalLabel, Chart
from common.rule import Rule
from common import log
from common.sample import sample
from common.logarithm import logadd
from common.logarithm import LOGZERO
from parser.parser import Parser
from parser_td.parser_td import ParserTD
from parser.vo_rule import VoRule
from parser_td.td_rule import TdRule
from collections import defaultdict
import math
import heapq
import StringIO 
import itertools

GRAPH_FORMAT = "hypergraph" 
STRING_FORMAT = "string"
TREE_FORMAT = "tree"

def parse_string(s):
    """
    Parse the RHS of a CFG rule.
    """
    tokens = s.strip().split()
    res = []
    nt_index = 0
    for t in tokens:
        if "$" in t: 
            new_token = NonterminalLabel.from_string(t)
            if not new_token.index:
                new_token.index = "_%i" % nt_index
                nt_index = nt_index + 1
        else: 
            new_token = t
        res.append(new_token)
    return res    
    
def _terminals_and_nts_from_string(string):
        terminals = set()
        nonterminals = set()
        for tok in string: 
            if isinstance(tok, NonterminalLabel):
                nonterminals.add(tok.label)
            else:
                terminals.add(tok)
        return terminals, nonterminals

class Grammar(dict):
    """
    Represents a set of rules as a mapping from rule IDs to rules and defines
    operations to be performed on the entire grammar.
    """

    def __init__(self, nodelabels = False, logprob = False):
        self.nodelabels = nodelabels  
        self.start_symbol = "truth" 
        self.logprob = logprob

        self.lhs_to_rules = defaultdict(set)
        self.nonterminal_to_rules = defaultdict(set) 
        self.rhs1_terminal_to_rules = defaultdict(set)
        self.rhs2_terminal_to_rules = defaultdict(set)
        self.startsymbol = None

    @classmethod
    def load_from_file(cls, in_file, rule_class = VoRule, reverse = False, nodelabels = False, logprob = False):
        """
        Loads a SHRG grammar from the given file. 
        See documentation for format details.
        
        rule_class specifies the type of rule to use. VoRule is a subclass using an arbitrary graph
        visit order (also used for strings). TdRule computes a tree decomposition on the first RHS
        when initialized.
        """

        output = Grammar(nodelabels = nodelabels, logprob = logprob)

        rule_count = 1
        line_count = 0
        is_synchronous = False

        rhs1_type = None
        rhs2_type = None

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
                        weight = 0.0 if not weights else (float(weights) if logprob else math.log(float(weights)))
                    except:
                        raise GrammarError, \
            "Line %i, Rule %i: Error near end of line." % (line_count, rule_count)
                   
                    try:  
                        lhs, rhsstring = content.split("->")
                    except:
                        raise GrammarError, \
            "Line %i, Rule %i: Invalid rule format." % (line_count, rule_count)
                    lhs = lhs.strip()
                    if rule_count == 1:
                        output.start_symbol = lhs
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
                    except (ParserError, IndexError), e: 
                        if rhs1_type == GRAPH_FORMAT:
                           raise ParserError,\
            "Line %i, Rule %i: Could not parse graph description: %s" % (line_count, rule_count, e.message)
                        else:
                           r1 = parse_string(rhs1) 
                           nts = [t for t in r1 if isinstance(t, NonterminalLabel)]
                           r1_nts = set([(ntlabel.label, ntlabel.index) for ntlabel in nts])
                           rhs1_type = STRING_FORMAT
  
                    if is_synchronous:
                        try:    # If the first graph in the file cannot be parsed, assume it's a string
                            if rhs2_type: 
                                assert rhs2_type == GRAPH_FORMAT
                            r2  = Hgraph.from_string(rhs2)
                            r2_nts = set([(ntlabel.label, ntlabel.index) for h, ntlabel, t in r2.nonterminal_edges()])
                            if not rhs2_type:
                                rhs2_type = GRAPH_FORMAT
                        except (ParserError, IndexError, AssertionError), e: 
                            if rhs2_type == GRAPH_FORMAT:
                               raise ParserError,\
                "Line %i, Rule %i: Could not parse graph description: %s" % (line_count, rule_count, e.message)
                            else:
                               r2 = parse_string(rhs2) 
                               nts = [t for t in r2 if isinstance(t, NonterminalLabel)]
                               r2_nts = set([(ntlabel.label, ntlabel.index) for ntlabel in nts])
                               rhs2_type = STRING_FORMAT

                        # Verify that nonterminals match up
                        if not r1_nts == r2_nts:
                            raise GrammarError, \
            "Line %i, Rule %i: Nonterminals do not match between RHSs: %s %s" % (line_count, rule_count, str(r1_nts), str(r2_nts))
                    else: 
                        r2 = None
                    try:    
                        if is_synchronous and reverse: 
                            output[rule_count] = rule_class(rule_count, lhs, weight, r2, r1, nodelabels = nodelabels, logprob = logprob)                                     
                        else: 
                            output[rule_count] = rule_class(rule_count, lhs, weight, r1, r2, nodelabels = nodelabels, logprob = logprob) 
                    except Exception, e:         
                        raise GrammarError, \
            "Line %i, Rule %i: Could not initialize rule. %s" % (line_count, rule_count, e.message)
                    buf = StringIO.StringIO() 
                    rule_count += 1

        output.is_synchronous = is_synchronous
        if is_synchronous and reverse:
            output.rhs1_type, output.rhs2_type = rhs2_type, rhs1_type
        else: 
            output.rhs1_type, output.rhs2_type = rhs1_type, rhs2_type

        output._compute_reachability_table_lookup()
        return output 

            
    def _compute_reachability_table_lookup(self):
        """
        Fill a table mapping rhs symbols to rules so that we can compute reachability.
        """
        for r in self:
            rule = self[r]
            if self.rhs1_type is GRAPH_FORMAT:
                self.lhs_to_rules[rule.symbol, len(rule.rhs1.external_nodes)].add(r)
                terminals, nonterminals = rule.rhs1.get_terminals_and_nonterminals(self.nodelabels)
                for nt in nonterminals:
                    self.nonterminal_to_rules[nt].add(r)
            elif self.rhs1_type is STRING_FORMAT:
               terminals, nonterminals = _terminals_and_nts_from_string(rule.rhs1) 
               self.lhs_to_rules[rule.symbol].add(r)
               for t in nonterminals: 
                   self.nonterminal_to_rules[t].add(r)

    def terminal_filter(self, input1, input2):

        input1_terminals = set()
        input2_terminals = set()

        if self.rhs1_type is GRAPH_FORMAT:
            input1_terminals, input1_nts = input1.get_terminals_and_nonterminals(self.nodelabels)
        elif self.rhs1_type is STRING_FORMAT:
            input1_terminals, input1_nonterminals = _terminals_and_nts_from_string(input1)
         
        if input2:      
            if self.rhs2_type is GRAPH_FORMAT:
                input2_terminals, input2_nts = input2.get_terminals_and_nonterminals(self.nodelabels)
            elif self.rhs2_type is STRING_FORMAT: 
                input2_terminals, input2_nts = _terminals_and_nts_from_string(input2)        

        accepted = list() 
        for r in self:
            terminals1, terminals2 = set(), set()
            if self.rhs1_type is GRAPH_FORMAT:
                terminals1, nonterminals = self[r].rhs1.get_terminals_and_nonterminals(self.nodelabels)
            elif self.rhs1_type is STRING_FORMAT:
                terminals1, nonterminals = _terminals_and_nts_from_string(self[r].rhs1)
            if input2:
                if self.rhs2_type is GRAPH_FORMAT:
                    terminals2, nonterminals = self[r].rhs2.get_terminals_and_nonterminals(self.nodelabels)
                elif self.rhs2_type is STRING_FORMAT:
                    terminals2, nonterminals = _terminals_and_nts_from_string(self[r].rhs2)
           
            if terminals1.issubset(input1_terminals):
                if input2 is None or terminals2.issubset(input2_terminals):
                    accepted.append(r)
            if not terminals1 and not terminals2:
                accepted.append(r) 

        return accepted    

   
    def reachable_rules(self, input1, input2):
        todo = list(self.terminal_filter(input1, input2))
        result = set()
        while todo:
            r = todo.pop()
            result.add(r)
            todo.extend(self.nonterminal_to_rules[self[r].symbol] - result)        
        return result 
 
    def normalize_by_groups(self, groups):
        """
        Normalize the grammar given a dictionary mapping rules to equivalence class ids.
        """
        norms = {}
        for r in self: 
            group = groups[r]
            if group in norms:
                norms[group] = logadd(norms[group], self[r].weight) 
            else:
                norms[group] = self[r].weight
        for r in self: 
            self[r].weight = self[r].weight - norms[groups[r]]
            

    def normalize_by_equiv(self, equiv):
        """
        Normalize the grammar so that all rules for which the function equiv returns an equivalent
        value sum up to 1. 
        """
        normalization_groups = {}
        for r in self:
            group = equiv(self[r])       
            normalization_groups[r] = group
        self.normalize_by_groups(normalization_groups)           
    
    def normalize_lhs(self):
        """
        Normalize the weights of the grammar so that all rules with the same LHS sum up to 1.
        """
        if self.rhs1_type == GRAPH_FORMAT:
            equiv = lambda rule: (rule.symbol, len(rule.rhs1.external_nodes))
        else: 
            equiv = lambda rule: rule.symbol
        self.normalize_by_equiv(equiv)
          
    def normalize_rhs1(self):
        """
        Normalize the weights of the grammar so that all rules with the same LHS and the same
        first RHS sum up to 1.
        """
        if isinstance(self[self.keys()[0]].rhs1, list):
            equiv = lambda rule: (rule.symbol, tuple(rule.rhs1))        
        else:
            equiv = lambda rule: (rule.symbol, rule.rhs1)        
        self.normalize_by_equiv(equiv, logprob)

    def normalize_rhs2(self):
        """
        Normalize the weights of the grammar so that all rules with the same LHS and the same
        second RHS sum up to 1.
        """
        if isinstance(self[self.keys()[0]].rhs2, list):
            equiv = lambda rule: (rule.symbol, tuple(rule.rhs2))        
        else:
            equiv = lambda rule: (rule.symbol, rule.rhs2)        
        self.normalize_by_equiv(equiv)

    def em_step(self, corpus, parser_class, normalization_groups, bitext = False):
        """ 
        Perform a single step of EM on the 
        """
        ll = 0.0

        counts = defaultdict(float)

        parser = parser_class(self)
        if bitext: 
            if parser_class == ParserTD:
                log.err("Bigraph parsing with tree decomposition based parser is not yet implemented. Use '-p basic'.")
                sys.exit(1)
            parse_generator = parser.parse_bitexts(corpus)
        else: 
            if self.rhs1_type == "string":
                if parser_class == ParserTD:
                    log.err("Parser class needs to be 'basic' to parse strings.")
                    sys.exit(1)
                else: 
                    parse_generator = parser.parse_strings(corpus)
            else: 
                parse_generator = parser.parse_graphs(corpus)
        
        i = 0
        for chart in parse_generator:
            i += 1   
            if not chart: 
                log.warn("No parse for sentence %d." % i)
                continue 
            inside_probs = chart.inside_scores()
            outside_probs = chart.outside_scores(inside_probs)
            ll += inside_probs["START"]
            counts_for_graph = chart.expected_rule_counts(inside_probs, outside_probs)
            for r in counts_for_graph:
                counts[r] = counts[r] + counts_for_graph[r]
      
        for r in counts: 
            if r in counts: 
                self[r].weight = counts[r]
            else: 
                self[r].weight = LOGZERO 
       
        self.normalize_by_groups(normalization_groups) 

        return ll 

    def em(self, corpus, iterations, parser_class = Parser, mode = "forward"):
        """
        Run EM training on the provided corpus for a given number of iterations.
        Mode can be "forward" (parse first RHS, normalize weights by 
        LHS + second RHS if any), or synchronous" (parse both sides at the same
        time, weights normalized by LHS only)
        """
        normalization_groups = {}
      
        if mode == "synchronous" or isinstance(corpus[0],tuple) :
            for r in self:             
                normalization_groups[r] = self[r].symbol
            bitext = True
        elif mode == "forward":
            if type(self[self.keys()[0]].rhs2) is list:
                for r in self:             
                    normalization_groups[r] = (self[r].symbol, tuple(self[r].rhs2))
            else:
                for r in self:
                    normalization_groups[r] = (self[r].symbol, self[r].rhs2) 
        
            bitext = False 
        self.normalize_by_groups(normalization_groups)

        for i in range(iterations):
            ll = self.em_step(corpus, parser_class, normalization_groups, bitext = bitext)
            log.info("Iteration %d, LL=%f" % (i, ll))


    def stochastically_generate(self):
        """
        Stochastically sample a derivation from this grammar.
        """

        def rec_choose_rules(nt):           
            if not nt in self.lhs_to_rules:
                raise DerivationException, "Could not find a rule for nonterminal %s with hyperedge tail type %d in grammar." % nt
            dist = [(self[r].weight, r) for r in self.lhs_to_rules[nt]]
            r = sample(dist)
            rule = self[r]
            dummy = DummyItem(rule)
            if self.rhs1_type == GRAPH_FORMAT:
                nt_edges = [((x[1].label, len(x[2])), x[1].index) for x in rule.rhs1.nonterminal_edges()]
            elif self.rhs1_type == STRING_FORMAT:
                nt_edges = [(x.label, x.index) for x in rule.rhs1 if isinstance(x, NonterminalLabel)]
            children = {} 
            prob = rule.weight 
            for edge in nt_edges:
                label, index = edge
                cweight, subtree = rec_choose_rules(label)
                prob += cweight
                if self.rhs1_type == GRAPH_FORMAT:  
                    nlabel, degree = label
                else:
                    nlabel = label
                children[(nlabel, index)] = subtree
            if children:
                new_tree = (dummy,children)
            else:
                new_tree = dummy
            return prob, new_tree
            
        firstrule = self[sorted(self.keys())[0]]
        
        if self.rhs1_type == GRAPH_FORMAT:
            start_symbol = firstrule.symbol, len(firstrule.rhs1.external_nodes)
        else:     
            start_symbol = firstrule.symbol
        prob, derivation = rec_choose_rules(start_symbol)
        return prob, derivation

    def kbest(self,k):
        """
        Produce k best derivations from this grammar.
        """
        # TODO: Document

        class KbestItem(object):
            
            def __init__(self, rule = None):
                if rule:
                    item = DummyItem(rule)
                    self.derivation = {"START":[("START",item)]}
                    self.weight = rule.weight
                    self.frontier = [item]

            def __lt__(self,other):
                return self.weight < other.weight 
            def __eq__(self,other):
                return self.weight == other.weight 
            def __gt__(self, other):
                return self.weight > other.weight      

            def spawn(self, grammar):
                """
                Take the next rule of the frontier, generate all possible derivations and return them. 
                """
                parent = self.frontier[0]
                rule = parent.rule

                if isinstance(rule.rhs1, Hgraph):
                    nt_edges = [((x[1].label, len(x[2])), x[1].index) for x in rule.rhs1.nonterminal_edges()]
                else:
                    nt_edges = [(x.label, x.index) for x in rule.rhs1 if isinstance(x, NonterminalLabel)]

                children = []               
                childlabels = []
                
                for edge in nt_edges:
                    label, index = edge
                    if isinstance(rule.rhs1, Hgraph):  
                        nlabel, degree = label
                    else:
                        nlabel = label                    
                    childlabels.append((nlabel,index))
                    children.append([(grammar[r].weight, DummyItem(grammar[r])) for r in grammar.lhs_to_rules[label]])

                if children: 
                    result = []
                    for combination in itertools.product(*children):
                        weights, items = zip(*combination)
                        new_kbest_item = KbestItem()
                        new_kbest_item.derivation = dict(self.derivation)
                        new_kbest_item.weight = self.weight + sum(weights)
                        new_kbest_item.frontier =  self.frontier[1:]
                        new_kbest_item.frontier.extend(items)
                        new_kbest_item.derivation[parent] = zip(childlabels, items)
                        result.append(new_kbest_item)
                    return result 
                else:            
                    self.frontier = self.frontier[1:]
                    self.derivation[parent] = []
                    return [self]
                    

        def convert_derivation(deriv, item):
            children = deriv[item]
            result = {}
            for edge, child in children:
                result[edge] = convert_derivation(deriv, child)       
            if result:
                return (item, result) 
            else: 
                return item
                                                
        kbest = []            
        heap = []
        
        firstrule = self[sorted(self.keys())[0]]
        if self.rhs1_type == GRAPH_FORMAT:
            start_symbol = firstrule.symbol, len(firstrule.rhs1.external_nodes)
        else:     
            start_symbol = firstrule.symbol

        for r in self.lhs_to_rules[start_symbol]:
            heapq.heappush(heap, KbestItem(self[r]))

        while True: 
            next_derivation = heapq.heappop(heap)

            if not next_derivation.frontier: # This is the next best complete derivation

                kbest.append((next_derivation.weight, convert_derivation(next_derivation.derivation, "START")[1]["START"]))
                continue
            if len(kbest) == k: # Are we done yet?
                return kbest
            
            for new in next_derivation.spawn(self):
                heapq.heappush(heap, new)

class DummyItem(object):
    """
    An simple chart item to keep track of rules used to generate a derivation
    from the grammar. As there is no input graph/string we do not need to 
    keep track of the covered span/subgraph.
    """
    def __init__(self, rule):
        self.rule = rule
