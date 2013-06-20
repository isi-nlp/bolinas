#!/usr/bin/env python2

import sys
import fileinput
from lib import log
from argparse import ArgumentParser
from config import config
from lib.hgraph.hgraph import Hgraph

#Import bolinas modules
from parser.parser import Parser
from parser.rule import Grammar, Rule
from parser_td.parser_td import ParserTD
from lib import output

if __name__ == "__main__":

    # Parse all the command line arguments, figure out what to do and dispatch to the appropriate modules. 
   
    # Initialize the command line argument parser 
    argparser = ArgumentParser(description = "Bolinas is a synchronous hyperedge replacement grammar toolkit. If no input file is provided the tool just verifies the grammar file and exits.")

    argparser.add_argument("grammar_file", help="A hyperedge replacement grammar (HRG) or synchronous HRG (SHRG).")
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
    argparser.add_argument("-m", "--weight_type", default="prob", help="Use real probabilities ('prob', default) or log probabilities ('logprob').")
    argparser.add_argument("-p","--parser", default="laut", help="Specify which parser to use. 'td': the tree decomposition parser of Chiang et al, ACL 2013 (default). 'laut' uses the Lautemann parser. 'cky' use a native CKY parser instead of the HRG parser if the input is a tree.")
    argparser.add_argument("-e","--edge_labels", action="store_true", default=False, help="Consider only edge labels when matching HRG rules. By default node labels need to match. Warning: The default is potentially unsafe when node-labels are used for non-leaf nodes in synchronous grammars.")
    argparser.add_argument("-bn","--boundary_nodes", action="store_true", help="Use the full edge representation for graph fragments instead of boundary node representation. This can provide some speedup for grammars with small rules.")
    argparser.add_argument("-s","--remove_spurious", default=False, action="store_true", help="Remove spurious ambiguity. Only keep the best derivation for identical derived objects.")
    argparser.add_argument("-v","--verbose", type=int, default=2, help="Stderr output verbosity: 0 (all off), 1 (warnings), 2 (info, default), 3 (details), 3 (debug)")
    
    args = argparser.parse_args()
    
    # If a configuration file was specified read in the configuration
    if args.config:
        config.load_config(file(args.config,'r'))
   

    # Otherwise just store configuration in the global configuration object
    config.__dict__.update(vars(args))

    # Verify command line parameters 
    if not config.output_type in ['forest', 'derivation', 'derived']:
        log.err("Output type (-ot) must be either 'forest', 'derivation', or 'derived'.")
        sys.exit(1)
    
    if not config.weight_type in ['prob', 'logprob']:
        log.err("Weight type (-m) must be either 'prob'or 'logprob'.")

    if config.output_type == "forest":

        if not config.output_file:       
            log.err("Need to provide '-o FILE_PREFIX' with output type 'forest'.")
            sys.exit(1)
        if config.k:
            log.warn("Ignoring -k command line option because output type is 'forest'.")    
    
    if not config.parser in ['td', 'laut', 'cky']:
        log.err("Parser (-p) must be either 'td', 'laut', or 'cky'.")
        sys.exit(1)

    if config.k > config.maxk:
        log.err("k must be <= than %i (defined in in config.py)." % config.maxk)
        sys.exit(1)

    if config.verbose < 0 or config.verbose > 4:
        log.err("Invalid verbosity level, must be 0-4.")
        sys.exit(1)
   

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

    # Run the selected parser                
    parser_class = {
             'laut': Parser,
             'td': None,
             'cky':None
             }[config.parser]

    with open(config.grammar_file,'ra') as grammar_file:
        grammar = Grammar.load_from_file(grammar_file, config.backward, nodelabels = (not config.edge_labels)) 
        log.info("Loaded %s%s grammar with %i rules."\
            % (grammar.rhs1_type, "-to-%s" % grammar.rhs2_type if grammar.rhs2_type else '', len(grammar)))
   
        parser = parser_class(grammar)
        if config.input_file:
            count = 1
            # Run the parser for each graph in the input
            for chart in parser.parse_graphs((Hgraph.from_string(x) for x in fileinput.input(config.input_file))):
                if config.output_type == "forest":
                    output_file = open("%s_%i.rtg" % (config.output_file, count), 'wa')
                    output_file.write(output.format_tiburon(chart))
                    output_file.close()
                    count = count + 1
                elif config.output_type == "derivation":                    
                    for score, derivation in chart.kbest('START', config.k, logprob = (config.weight_type == "logprob")):
                        output_file.write("%s\t#%f\n" % (output.format_derivation(derivation), score))
                    output_file.write("\n")
                elif config.output_type == "derived":
                    for score, derivation in chart.kbest('START', config.k, logprob = (config.weight_type == "logprob")):
                        output_file.write("%s\t#%f\n" % (output.apply_derivation(derivation).to_string(newline = False), score))
                    output_file.write("\n")




