#!/usr/bin/env python2

import sys
import pprint
from argparse import ArgumentParser
from config import config

#Import bolinas modules
from parser.parser import Parser
from parser.parser import chart_to_tiburon
from parser.rule import Grammar, Rule
from parser_td.parser_td import ParserTD
from lib.amr.amr import Amr

def die(msg):
    print >>sys.stderr, msg
    sys.exit(1)

if __name__ == "__main__":
   
    # Initialize the command line argument parser 
    argparser = ArgumentParser(description = "Bolinas is a synchronous hyperedge replacement grammar toolkit. If no input file is provided the tool just verifies the grammar file and exits.")

    argparser.add_argument("grammar_file", help="A hyperedge replacement grammar (HRG) or synchronouse HRG (SHRG).")
    argparser.add_argument("input_file", nargs="?", help="Input file containing one object per line or pairs of objects. Use - to read from stdin.")
    argparser.add_argument("-c", "--config", help="Read configuration file.")
    argparser.add_argument("-o","--output_file", type=str, help="Write output to a file instead of stdout.")
    direction = argparser.add_mutually_exclusive_group()
    direction.add_argument("-f","--forward", action="store_true", default=True, help="Apply the synchronous HRG left-to-right (default)")
    direction.add_argument("-r","--backward", action="store_true", default=False, help="Apply the synchronous HRG right-to-left.")
    direction.add_argument("-b","--bitext", action="store_true", default=False, help="Parse pairs of objects from the input file (default with -t).")
    argparser.add_argument("-ot","--output_type", type=str, default="derived", help="Set the type of the output to be produced for each object in the input file. \n'forest' produces parse forests.\n'derivation' produces k-best derivations.\n'derived' produces k-best derived objects (default).")
    mode = argparser.add_mutually_exclusive_group()
    mode.add_argument("-g",type=int, help ="Generate G random derivations from the grammar stochastically. Cannot be used with -k.")
    mode.add_argument("-k",type=int, default=1, help ="Generate K best derivations for the objects in the input file. Cannot be used with -g (default with K=1).")
    weights = argparser.add_mutually_exclusive_group()
    weights.add_argument("-d","--randomize", default=False, action="store_true", help="Randomize weights to be distributed between 0.2 and 0.8. Useful for EM training.")
    weights.add_argument("-n","--normalize", default=False, action="store_true", help="Normalize weights to sum to 1.0 for all rules with the same LHS.") 
    weights.add_argument("-t","--train", default=False, action="store_true", help="Use EM to train weights for the grammar using the input. Initialize with the weights in the grammar file or random weights if none are provided.")
    argparser.add_argument("-s","--remove_spurious", default=False, action="store_true", help="Remove spurious ambiguity. Only keep the best derivation for identical derived objects.")
    argparser.add_argument("-p","--parser", default="laut", help="Specify which parser to use. 'td': the tree decomposition parser of Chiang et al, ACL 2013 (default). 'laut' uses the Lautemann parser. 'cky' use a native CKY parser instead of the HRG parser if the input is a tree.")
    argparser.add_argument("-bn","--boundary_nodes", action="store_true", help="Use the full edge representation for graph fragments instead of boundary node representation. This can provide some speedup for grammars with small rules.")
    
    args = argparser.parse_args()

    # Verify command line parameters 
    if not args.output_type in ['forest', 'derivation', 'derived']:
        die("Output type (-ot) must be either 'forest', 'derivation', or 'derived'.")
    
    if not args.parser in ['td', 'laut', 'cky']:
        die("Parser (-p) must be either 'td', 'laut', or 'cky'.")

    if args.k > config.maxk:
        die("k must be <= than %i (defined in in config.py)." % config.maxk)

    # If a configuration file was specified read in the configuration
    if args.config:
        config.load_config(file(args.config,'r'))
   
    # Otherwise just store configuration in the global configuration object
    config.__dict__.update(vars(args))

    if config.parser == "laut":
        # Run the non-TD HRG parser 
        grammar = Grammar.load_from_file(file(config.grammar_file,'ra'), config.backward)                
        print >>sys.stderr, "Loaded %s%s grammar with %i rules."\
             % (grammar.rhs1_type, "-to-%s" % grammar.rhs2_type if grammar.rhs2_type else '', len(grammar))

        parser = Parser(grammar)
        if config.input_file:
            with file(config.input_file,'r') as in_file:
                for chart in parser.parse_graphs((Amr.from_string(x) for x in in_file)):
                    print chart 

    elif config.parser == "td":
        # Run the tree decomposition HRG parser
        pass
    elif config.parser == "cky":
        # Run native CKY parser for TSG
        pass
    
