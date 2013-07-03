#!/usr/bin/env python2

import sys
import fileinput
from config import config
from lib import log
from argparse import ArgumentParser
from lib.hgraph.hgraph import Hgraph
from lib.exceptions import DerivationException

from common.grammar import Grammar

#Import bolinas modules
from parser.parser import Parser
from parser.vo_rule import VoRule
from parser_td.td_rule import TdRule
from parser_td.td_item import Item
from parser_td.parser_td import ParserTD
from lib import output

if __name__ == "__main__":

    # Parse all the command line arguments, figure out what to do and dispatch to the appropriate modules. 
   
    # Initialize the command line argument parser 
    argparser = ArgumentParser(description = "Bolinas is a toolkit for synchronous hyperedge replacement grammars.")

    argparser.add_argument("grammar_file", help="A hyperedge replacement grammar (HRG) or synchronous HRG (SHRG).")
    argparser.add_argument("input_file", nargs="?", help="Input file containing one object per line or pairs of objects. Use - to read from stdin.")
    argparser.add_argument("-o","--output_file", type=str, help="Write output to a file instead of stdout.")
    direction = argparser.add_mutually_exclusive_group()
    direction.add_argument("-f","--forward", action="store_true", default=True, help="Apply the synchronous HRG left-to-right (default)")
    direction.add_argument("-r","--backward", action="store_true", default=False, help="Apply the synchronous HRG right-to-left.")
    direction.add_argument("-b","--bitext", action="store_true", default=False, help="Parse pairs of objects from an input file with alternating lines.")
    argparser.add_argument("-ot","--output_type", type=str, default="derived", help="Set the type of the output to be produced for each object in the input file. \n'forest' produces parse forests.\n'derivation' produces k-best derivations.\n'derived' produces k-best derived objects (default).")
    mode = argparser.add_mutually_exclusive_group()
    mode.add_argument("-g",type=int, help ="Generate G random derivations from the grammar stochastically. Cannot be used with -k.")
    mode.add_argument("-k",type=int, default=1, help ="Generate K best derivations for the objects in the input file. Cannot be used with -g (default with K=1).")
    weights = argparser.add_mutually_exclusive_group()
    weights.add_argument("-d","--randomize", default=False, action="store_true", help="Randomize weights to be distributed between 0.2 and 0.8. Useful for EM training.")
    weights.add_argument("-n","--normalize", default=False, action="store_true", help="Normalize weights to sum to 1.0 for all rules with the same LHS.") 
    weights.add_argument("-t","--train", default=5, help="Use TRAIN iterations of EM to train weights for the grammar using the input (graph, string, or pairs of objects in alternating lines). Initialize with the weights in the grammar file or with uniform weights if none are provided. Writes a grammar file with trained weights to the output.")
    argparser.add_argument("-m", "--weight_type", default="prob", help="Use real probabilities ('prob', default) or log probabilities ('logprob').")
    argparser.add_argument("-p","--parser", default="basic", help="Specify which graph parser to use. 'td': the tree decomposition parser of Chiang et al, ACL 2013 (default). 'basic': a basic generalization of CKY that matches rules according to an arbitrary visit order on edges (less efficient).")
    argparser.add_argument("-e","--edge_labels", action="store_true", default=False, help="Consider only edge labels when matching HRG rules. By default node labels need to match. Warning: The default is potentially unsafe when node-labels are used for non-leaf nodes on the target side of a synchronous grammar.")
    argparser.add_argument("-bn","--boundary_nodes", action="store_true", help="In the tree decomposition parser, use the full representation for graph fragments instead of the compact boundary node representation. This can provide some speedup for grammars with small rules.")
    argparser.add_argument("-s","--remove_spurious", default=False, action="store_true", help="Remove spurious ambiguity. Only keep the best derivation for identical derived objects.")
    argparser.add_argument("-v","--verbose", type=int, default=2, help="Stderr output verbosity: 0 (all off), 1 (warnings), 2 (info, default), 3 (details), 3 (debug)")
    
    args = argparser.parse_args()
    
    # Verify command line parameters 
    if not args.output_type in ['forest', 'derivation', 'derived']:
        log.err("Output type (-ot) must be either 'forest', 'derivation', or 'derived'.")
        sys.exit(1)
    
    if not args.weight_type in ['prob', 'logprob']:
        log.err("Weight type (-m) must be either 'prob'or 'logprob'.")

    if args.output_type == "forest":
        if not args.output_file:       
            log.err("Need to provide '-o FILE_PREFIX' with output type 'forest'.")
            sys.exit(1)
        if args.k:
            log.warn("Ignoring -k command line option because output type is 'forest'.")    
    
    if not args.parser in ['td', 'basic']:
        log.err("Parser (-p) must be either 'td' or 'basic'.")
        sys.exit(1)
    
    if args.parser != 'td' and args.boundary_nodes: 
        log.warn('The -bn option is only relevant for the tree decomposition parser ("-p td").')

    if args.k > config.maxk:
        log.err("k must be <= than %i (defined in in args.py)." % args.maxk)
        sys.exit(1)

    if args.verbose < 0 or args.verbose > 4:
        log.err("Invalid verbosity level, must be 0-4.")
        sys.exit(1)
  

    # Updat global configuration with command line args 
    config.__dict__.update(vars(args))

    # Definition of verbosity levels 
    log.LOG = {0:{log.err},
               1:{log.err, log.warn},
               2:{log.err, log.warn, log.info},
               3:{log.err, log.warn, log.info, log.chatter},
               4:{log.err, log.warn, log.chatter, log.info, log.debug}
              }[config.verbose]
    
    # Direct output to stdout if no filename is provided
    if config.output_type is not "derivation":
        if config.output_file:
            output_file = open(config.output_file,'wa')
        else:
            output_file = sys.stdout        

    with open(config.grammar_file,'ra') as grammar_file:
        # Select the parser and rule class to use 
        if config.parser == 'td':
            parser_class = ParserTD 
            rule_class = TdRule
            if config.boundary_nodes:
                parser_class.item_class = Item

        elif config.parser == 'basic':
            parser_class = Parser
            rule_class = VoRule

        # Read the grammar
        grammar = Grammar.load_from_file(grammar_file, rule_class, config.backward, nodelabels = (not config.edge_labels)) 
        if len(grammar) == 0:
            log.err("Unable to load grammar from file.")
            sys.exit(1)

        log.info("Loaded %s%s grammar with %i rules."\
            % (grammar.rhs1_type, "-to-%s" % grammar.rhs2_type if grammar.rhs2_type else '', len(grammar)))
 
        if grammar.rhs2_type is None and config.output_type == "derived":
            log.err("Can only build derived objects (-ot derived) with synchronous grammars.") 
            sys.exit(1)
         
        parser = parser_class(grammar)

        if grammar.rhs1_type == "string":
            if parser_class == "td":
                log.err("Parser class needs to be 'basic' to parse strings.")
                sys.exit(1)
            else: 
                parse_generator = parser.parse_strings(x.strip().split() for x in fileinput.input(config.input_file))
        else: 
            parse_generator = parser.parse_graphs(Hgraph.from_string(x) for x in fileinput.input(config.input_file))
        
        # Process input (if any) and produce desired output 
        if config.input_file:
            count = 1
            # Run the parser for each graph in the input
            for chart in parse_generator:
                # Produce Tiburon format derivation forests
                if config.output_type == "forest":
                    output_file = open("%s_%i.rtg" % (config.output_file, count), 'wa')
                    output_file.write(output.format_tiburon(chart))
                    output_file.close()
                    count = count + 1

                # Produce k-best derivations
                elif config.output_type == "derivation":                    
                    for score, derivation in chart.kbest('START', config.k, logprob = (config.weight_type == "logprob")):
                        output_file.write("%s\t#%f\n" % (output.format_derivation(derivation), score))
                    output_file.write("\n")

                # Produce k-best derived graphs/strings
                elif config.output_type == "derived":
                    if grammar.rhs2_type == "hypergraph":
                        for score, derivation in chart.kbest('START', config.k, logprob = (config.weight_type == "logprob")):
                            try:
                                output_file.write("%s\t#%f\n" % (output.apply_graph_derivation(derivation).to_string(newline = False), score))
                            except DerivationException:
                                log.err("Derivation produces contradicatory node labels in derived graph. Skipping.")
                    elif grammar.rhs2_type == "string":
                        for score, derivation in chart.kbest('START', config.k, logprob = (config.weight_type == "logprob")):
                            try:
                                output_file.write("%s\t#%f\n" % (" ".join(output.apply_string_derivation(derivation)), score))
                            except DerivationException:
                                log.err("Derivation produces contradicatory node labels in derived graph. Skipping.")
        
                    output_file.write("\n")




