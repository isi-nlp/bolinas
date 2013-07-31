'''
Methods to format Bolinas output.
'''

from common.hgraph.hgraph import Hgraph
from common.cfg import NonterminalLabel
from common import log 
 
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
        graph =  item.rule.rhs2.clone_canonical(prefix = str(step[0]))
        step[0] = step[0] + 1
        return graph

    def combiner(item, childobjs):
        graph = leaf(item)
        for nt, cgraph in childobjs.items():
                p,r,c = graph.find_nt_edge(*nt)
                fragment = Hgraph.from_triples([(p,r,c)],graph.node_to_concepts)
                graph = graph.replace_fragment(fragment, cgraph)
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

def format_tiburon_old(chart):
    """
    Return a tiburon format RTG that describes a parse forest. 
    """
    import pprint
    pprint.pprint(chart) 
    lines = ["_START"]
    for item in chart:
        count = 0
        for possibility in chart[item]: 
            for nt,child in possibility.items():                
                if nt == "START":
                    parent_rtg_state = "_START" 
                else:
                    symbol, index = nt                 
                    parent_rtg_state = item.uniq_str()
                if child in chart: 
                    for cpossibility in chart[child]: 
                        childstrl = []
                        for cnt, citem in cpossibility.items():
                            if citem in chart:
                                childstrl.append(citem.uniq_str())
                            else:
                                childstrl.append("R%s" % citem.rule.rule_id)
                        if nt == "START":
                            childstr = ("R%s(%s)") % (child.rule.rule_id, " ".join(childstrl))
                        else: 
                            childstr = ("R%s_%s%s(%s)" % (child.rule.rule_id, symbol, index, " ".join(childstrl)))
                        print childstr
                        lines.append("%s -> %s\t#%f" % (parent_rtg_state, childstr, child.rule.weight))
                    

                else:  
                    childstr = "R%s" % child.rule.rule_id
                    lines.append("%s -> %s\t#%f" % (parent_rtg_state, childstr, child.rule.weight))
    return "\n".join(lines)

def format_tiburon(chart):
    """
    Return a tiburon format RTG that describes a parse forest. 
    """
    lines = ["_START"]
    for item in chart:
        for possibility in chart[item]: 

            childstrl = []
            if item == "START":
                parent_rtg_state = "_START"
                for nt,child in possibility.items():
                    if not child in chart: 
                        lines.append("_START -> %s\t#%f" % (child.rule.rule_id, 1.0))
                    else:
                        lines.append("_START -> %s\t#%f" % (child.uniq_str(), 1.0))
            else: 
                parent_rtg_state = item.uniq_str()
                for nt,child in possibility.items():                
                    symbol, index = nt                 
                    if not child in chart: 
                        childstrl.append("%s$%s(%d)" % (symbol, index, child.rule.rule_id))
                    else:
                        childstrl.append("%s$%s(%s)" % (symbol, index, child.uniq_str()))
                
                childstr = "%d(%s)" % (item.rule.rule_id, " ".join(childstrl))
                lines.append("%s -> %s\t#%f" % (parent_rtg_state, childstr, item.rule.weight))
                    

    return "\n".join(lines)


