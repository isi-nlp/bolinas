'''
Apply the rules encoded in a HRG derivation tree to construct a graph. 

@author: Daniel Bauer (dbauer)
@since: 2012-07-6
'''

from lib.pyparsing import Literal,Word,CharsNotIn, OneOrMore, ZeroOrMore, Forward, nums, alphas, Optional 
from lib.amr.dag import Dag
from collections import defaultdict
import re
import sys
import copy
import cPickle
import argparse

def make_bracketed_parser():
    """
    Pyparsing parser for trees as produced by Tiburon. 
    Returns the tree as a nested structure of (head, [children]) tuples. 
    """
    lpar  = Literal( "(" ).suppress()
    rpar  = Literal( ")" ).suppress()

    node_name =  Word(alphas+nums+"_@")  
    
    expr = Forward()
    rels = Word(alphas+nums+"-_@") + lpar + expr + rpar
    rels.setParseAction(lambda s, loc, tok: (tok[0], tok[1:]))   

    expr << (node_name + Optional(lpar + OneOrMore(rels) + rpar))

    return rels 


def parse_hgraph_file(amr_file):
    """
    Read in an .amr file and produce a dictionary of amr fragments index by rule IDs. 
    
    @param amr_file: The file object to read from.
    """
    result = {}

    for line in amr_file:
        rulename, amr_s = line.split(',',1)
        amr = Dag.from_string(amr_s)
        result[rulename] = amr 
    return result


def build_derived_hgraph(derivation_tree, grammar):
    """
    Construct an AMR from a derivation tree. 
    
    @param derivation_tree:
        a given derivationt tree as returned by 
        L{make_bracketed_parser}. 

    @param grammar: 
        a dictionary of AMR fragments indexed by rule IDs
        as returned by L{parse_hgraph_file}

    @return: a L{amr.amr.Dag} object.
    """

    dtree = dict(list(derivation_tree)) 
    
    assert len(dtree)==1   # Derivation tree must be single-rooted
    nt_str = dtree.keys()[0]

    step = [0] # Wrapped in a list so we can preserve the closure on write.
    def rec_substitute(dtree):
        """
        Traverse the derivation tree and perform substitutions in the resulting graph. 
        """
        graph = grammar[dtree[0]].clone_canonical(prefix = str(step[0]))
        step[0] += 1

        subtree = dict(dtree[1:])
        for nt in subtree:
            # Find edge to replace
            edge = None
            for p,r,c in graph.nonterminal_edges():
                if nt == r.label:
                    edge = (p,r,c)
                    break
            if not edge: 
                sys.stderr.write("No nonterminal %s found.\n" % nt.label) 
                return Dag()
            fragment = Dag.from_triples([edge])                               
            new_graph = rec_substitute(subtree[nt])                                           
            graph = graph.replace_fragment(fragment, new_graph) 

        return graph 
        
    return rec_substitute(dtree[nt_str]) 

def main():
    
    argparser = argparse.ArgumentParser(description='Apply Hyperedge Replacement Grammar (HRG) Derivations to Construct Hypergraphs.')
    argparser.add_argument('grammar_file', help='File containing the graph fragment RHSs of a HRG.')
    argparser.add_argument('dtree_file', help='Input file contains one derivation tree per line. - for stdin.')
    argparser.add_argument('-f', 
          help="Format of the derivation trees. 'tiburon' (default) for Tiburon tree format, e.g. 'root(child1 child2)' or 'prefix' for lisp style notation, e.g. '(root child1 child2)'.", default="tiburon", dest="format", metavar="TREEFORMAT") 
    argparser.add_argument('-o', 
          help="Output file. '-' for stdout (default).", default="-", dest="output_file", metavar="FILENAME") 


    (options, args) = argparser.parse_args()

    format = options.format
    if format not in ["tiburon","prefix"]:
        print >>sys.stderr, argparser.get_usage() 
        sys.exit(1)

    parser = make_bracketed_parser()

    try:
        grammar_file = open(args.grammar_file,'r')
    except:
        print >>sys.stderr, "ERROR: Could not open grammar RHS file %s for reading." % args.grammar_file 

    try: #First attempt to unpickle.
        amrs = cPickle.load(grammar_file) 
    except cPickle.UnpicklingError:    
        try: #  Otherwise load the plain hypergraph fragment file. 
            amrs = parse_hgraph_file(amr_file) 
        except:
            print >>sys.stderr, "ERROR: Could not parse grammar RHS file %s. Please check format." % args.grammar_file
            sys.exit(1)

    if args.dtree_file == "-":
        in_file = sys.stdin
    else:    
        try:
            in_file = open(args.dtree_file,'r')
        except:    
            print >>sys.stderr, "ERROR: Could not open derivation tree file %s for reading." % args.grammar_file 
            sys.exit(1)

    if options.out_file == "-":
        out_file = sys.stdout
    else: 
        try:
            out_file = open(options.output_file,'w')
        except:
            print >>sys.stderr, "ERROR: Could not open output file %s for writing." % options.out_file    

    for line in in_file:
        if line.strip() == '0' or ':' in line:
            print >>out_file, "# Empty derivation."
        else:    
            parse = parser.parseString(line.strip())
            try:
                result =  build_derived_hgraph(parse, amrs)  
                if '#' in result:
                    raise Exception("incomplete")
                print >>out_file, result.to_string(newline = False)
            except: 
                print >>out_file, "# Could not build terminal hypergraph from derivation."

if __name__ == "__main__":
    main()
