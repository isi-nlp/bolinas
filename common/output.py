'''
Methods to format Bolinas output.
'''

from common.hgraph.hgraph import Hgraph
from common.cfg import NonterminalLabel
from common import log 
from common.exceptions import DerivationException
import math
 
def walk_derivation(derivation, combiner, leaf):
    """
    Traverse a derivation as returned by parser.item.Chart.kbest. Apply combiner to 
    a chart Item and a dictionary mapping nonterminals (and indices) to the result of 
    all child items.
    """
    if type(derivation) is not tuple: 
        if derivation == "START":
            return None
        return leaf(derivation)
    else: 
        item, children = derivation[0], derivation[1]        
        childobjs = dict([(rel, walk_derivation(c, combiner, leaf)) for (rel, c) in children.items()])
       
        if item == "START":
            return childobjs["START"]
       
        return combiner(item, childobjs)

def format_derivation(derivation):
    """
    Output the derivation as a tree in Lisp/PTB format
    e.g. (A B C) is a tree with root A and ordered children 
    B and C.
    """
    def combiner(item, childobjs):
        children = []
        for nt, child in childobjs.items():
            edgestring = "$".join(nt)
            children.append("%s(%s)" % (edgestring, child))
        childstr = " ".join(children)
        return "%s(%s)" % (item.rule.rule_id,  childstr)
   
    def leaf(item):
        return str(item.rule.rule_id) 
        
    return walk_derivation(derivation, combiner, leaf)

def apply_graph_derivation(derivation):
    """
    Assemble a derived graph by executing the operations specified in the derivation (as produced by 
    parser.item.Chart.kbest)
    """
    step = [0] # Wrap in list so that we preserve closure on write.

    def leaf(item):
        if item.rule.rhs2:
            graph =  item.rule.rhs2.clone_canonical(prefix = str(step[0]))
        else:
            graph =  item.rule.rhs1.clone_canonical(prefix = str(step[0]))
        step[0] = step[0] + 1
        return graph

    def combiner(item, childobjs):
        graph = leaf(item)
        for nt, cgraph in childobjs.items():
                p,r,c = graph.find_nt_edge(*nt)
                fragment = Hgraph.from_triples([(p,r,c)],graph.node_to_concepts)
                try:
                    graph = graph.replace_fragment(fragment, cgraph)
                except AssertionError, e:
                    raise DerivationException, "Incompatible hyperedge type for nonterminal %s." % str(nt[0])
        step[0] = step[0] + 1
        return graph 

    return walk_derivation(derivation, combiner, leaf) 

def apply_string_derivation(derivation):
    """
    Assemble a derived string by executing the operations specified in the derivation (as produced by 
    parser.item.Chart.kbest)
    """
    
    def leaf(item):
        return item.rule.rhs2
    
    def combiner(item, childobjs):
        result = []
        for t in item.rule.rhs2:
            if isinstance(t, NonterminalLabel):
                result.extend(childobjs[t.label, t.index])
            else: 
                result.append(t)
        return result
         
    return walk_derivation(derivation, combiner, leaf)

def format_tiburon(chart, logprob = False):
    """
    Return a tiburon format RTG that describes a parse forest. 
    """
    lines = set()
    for item in chart:
        for possibility in chart[item]: 

            childstrl = []
            if item == "START":
                parent_rtg_state = "_START"
                for nt,child in possibility.items():
                    lines.add("_START -> %s\t#%f" % (child.uniq_str(), 0.0 if logprob else 1.0))
                    if not child in chart: 
                        lines.add("%s -> %s\t#%f" % (child.uniq_str(), child.rule.rule_id, child.rule.weight))
            else: 
                parent_rtg_state = item.uniq_str()
                for nt,child in possibility.items():                
                    symbol, index = nt                 
                    if not child in chart: 
                        childstrl.append("%s$%s(%s)" % (symbol, index, child.uniq_str()))
                        weight = child.rule.weight if logprob else math.exp(child.rule.weight)
                        lines.add("%s -> %s\t#%f" % (child.uniq_str(), child.rule.rule_id, weight))
                    else:
                        childstrl.append("%s$%s(%s)" % (symbol, index, child.uniq_str()))
                
                childstr = "%d(%s)" % (item.rule.rule_id, " ".join(childstrl))
                
                weight = item.rule.weight if logprob else math.exp(item.rule.weight)
                lines.add("%s -> %s\t#%f" % (parent_rtg_state, childstr, weight))
                    

    linelist = ["_START"] + sorted(list(lines))
    return "\n".join(linelist)


