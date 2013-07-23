#!/usr/bin/env python2
from hgraph import Hgraph
from collections import defaultdict
import itertools
import math
import sys
import random
import copy
from lib import pyparsing
import timeit

from optparse import OptionParser, OptionGroup

"""
An implementation of the SMATCH metric for Feature Structure Graphs (such as AMRs).

@author: Daniel Bauer (bauer@cs.columbia.edu) 
@since: 2012-08-17
"""

def get_mappings(l2, l1):
     if len(l2) >= len(l1):
         return (zip(t,l1) for t in itertools.permutations(l2, len(l1)))
     else:
         return (zip(l2,l) for l in itertools.permutations(l1, len(l2)))

def get_random_start(l2,l1):
     if len(l2) >= len(l1):
         random.shuffle(l1)
         return zip(l2,l1)        
     else:
         random.shuffle(l2)
         return zip(l2, l1)

def compute_score(triples1, triples2):
    """
    Compute precision, recall, and f-score. Variable names must be identical.
    """
    t1 = set(triples1)
    t2 = set(triples2)
    prec = len(t1.intersection(t2)) / float(len(t2))
    rec = len(t1.intersection(t2)) / float(len(t1))
    if prec == 0.0 and rec == 0.0:
        return 0.0, 0.0, 0.0 
    f = 2 * prec * rec  / (prec + rec)
    return prec, rec, f

def get_smatch_triples(amr):

    t = []
    for tr in amr.triples(instances = True):
        if tr[1].lower().endswith("-of"):
            p,r,ch = tr
            t.append((ch[0], r[:-3], (p,)))
        else:
            t.append(tr)


    for r in amr.roots: 
        t.append((r,"TOP",amr.node_to_concepts[r]))
    return t

def compute_score_and_matching_vars(amr1, t2, mapping):
    """
    Compute precision, recall, and f-score. Variable names must be identical.
    """
    inv_map = dict((v,k) for (k,v) in mapping.items())

    t1 = map_triples(amr1, mapping)
    #test_amr = amr1.apply_node_map(mapping)
    # 
    #t1 = set(test_amr.triples())
    #for r in test_amr.roots: 
    #    t1.add((r,"TOP",test_amr.node_to_concepts[r]))
    #t2 = set(amr2.triples())
    #for r in amr2.roots: 
    #    t2.add((r,"TOP",amr2.node_to_concepts[r]))

    common = t1.intersection(t2)

    prec = len(common) / float(len(t1))
    rec = len(common) / float(len(t2))
    if prec == 0.0 and rec == 0.0:
        return 0.0, 0.0, 0.0, []
    f = 2 * prec * rec  / (prec + rec)

    matching = set()
    for p,r,c in common:
        if p in inv_map: 
            matching.add((inv_map[p],p))        
        if type(c) is tuple:
            for child in c: 
                if child in inv_map:
                    matching.add((inv_map[child],child))
            else:
                if c in inv_map:
                    matching.add((inv_map[c],c))

    return prec, rec, f, matching
    

def compute_smatch_precise(amr1, amr2):
    """
    Do not ever call this function for any realistically large AMRs!
    """
    nodes1 = amr1.get_nodes()
    nodes2 = amr2.get_nodes()
    # map nodes1 to nodes2
    best_f = 0.0
    prec_with_best_f = 0.0
    rec_with_best_f = 0.0
    mappings = list(get_mappings(nodes1, nodes2))
    amr1trips = get_smatch_triples(amr1)
    amr2trips = get_smatch_triples(amr2)
    for mapping_tuples in mappings: 
        mapping = dict(mapping_tuples)
        new_amr1 = map_triples(amr1trips, mapping)
        prec, rec, f = compute_score(new_amr1, amr2trips)
        if f >= best_f:
            best_f = f
            prec_with_best_f = prec
            rec_with_best_f = rec
    return prec_with_best_f, rec_with_best_f, best_f

def get_parallel_start(amr1, amr2):       
    """
    Assume that the two AMRs line up perfectly. This is obviously super fast
    for correct AMRs.
    """
    return zip(nodes1.get_nodes(), nodes2.get_nodes())

def get_concept_match_start(amr1, amr2):
    """
    Assume that variables can only match up when they have the same 
    concept and choose a random alignment.
    """
    concept_to_nodes1 = defaultdict(list)
    for n, c in amr1.node_to_concepts.items():
        concept_to_nodes1[c].append(n)
    
    concept_to_nodes2 = defaultdict(list)
    for n, c in amr2.node_to_concepts.items():
        concept_to_nodes2[c].append(n)
   
    res = []
    for match in set(concept_to_nodes1).intersection(concept_to_nodes2):
        res.append((random.choice(concept_to_nodes1[match]), random.choice(concept_to_nodes2[match])))
    return res

def get_root_align_start(amr1, amr2):
    """
    Align the root nodes, then get a random mapping.
    """
    nodes1 = [n for n in amr1.get_nodes() if not n in amr1.roots]
    nodes2 = [n for n in amr2.get_nodes() if not n in amr1.roots]

    return get_random_start(nodes1, nodes2) + [(amr1.roots[0], amr2.roots[0])]
        

def map_triples(triples, map):
    res = []
    for p,rel,child in triples:
        new_p = map[p] if p in map else p
        if type(child) is tuple:
            new_c = tuple([map[c] if c in map else c for c in child])
        else:
            new_c = map[child] if child in map else child    
        res.append((new_p, rel, new_c))
    return set(res)

def compute_smatch_hill_climbing(amr1in, amr2in, starts = 10, method = get_random_start, restart_threshold = 1.0):       

    """
    Run hill climbing search in the space of variable mappings to find the smatch score between two AMRs.

    >>> amr1 = Hgraph.from_string("(a / amr-unknown :domain-of (x1 / population-quantity) :quant-of (x0 / people :loc-of (b / state :name (x2 / name :op0 (washington / washington) ))))")
    >>> amr2 = Hgraph.from_string("(t / amr-unknown :domain-of (x11 / density-quantity) :loc (x60 / state :name (x13 / name :op0 (x12 / washington) )))")
    >>> compute_smatch_hill_climbing(amr1,amr2, starts = 10) 
    (0.6666666666666666, 0.8, 0.7272727272727272)
    """
        
    amr1 = amr1in.clone_canonical(prefix="t")
    amr2 = amr2in.clone_canonical(prefix="g")

    best_f = -1.0 
    prec_with_best_f = 0.0
    rec_with_best_f = 0.0
    nodes1 = amr1.get_nodes()
    nodes2 = amr2.get_nodes()
    best_mapping =  {}

    for i in range(starts):
        mapping_tuples = method(amr1, amr2)    
        mapping = dict(mapping_tuples)
        amr1trips = get_smatch_triples(amr1)
        amr2trips = get_smatch_triples(amr2)
        max_prec, max_rec, max_f, matching_tuples = compute_score_and_matching_vars(amr1trips, amr2trips, mapping)
        prev_f = -1.0

        while max_f > prev_f:
            prev_f = max_f
            matching_dict = dict(matching_tuples)
            left_in_a1 = [n for n in nodes1 if not n in matching_dict.keys()]
            left_in_a2 = [n for n in nodes2 if not n in matching_dict.values()]           

            # Do hill climbing step
            for x in left_in_a1: # Explore all neighbors                    
                for y in left_in_a2:        
                    try_mapping = copy.copy(matching_dict)
                    try_mapping[x] = y 
                    prec, rec, f, try_matching_tuples = compute_score_and_matching_vars(amr1trips,amr2trips, try_mapping)
                    if f > max_f: 
                        max_prec = prec
                        max_rec = rec
                        matching_tuples = try_matching_tuples
                        mapping = try_mapping
                        max_f = f

        if prev_f > best_f:
                best_f = prev_f
                prec_with_best_f = max_prec
                rec_with_best_f = max_rec
                best_mapping = mapping 

        if best_f > restart_threshold: # If we have reached the threshold after this start, just return the result 
            break        

    return prec_with_best_f, rec_with_best_f, best_f

def mean(l, emptylines = 0):
    if len(l)==0:           
        return float("NaN")
    else:
        sumval, normalizer = 0, 0
        for val, this_l in l: 
            sumval += val * this_l
            normalizer += this_l 

        return sumval / normalizer #(len(l) + emptylines)

def compute_smatch_batch(gold_filename, test_filename, starts, method ,
                         restart_threshold, concept_edges, precise, 
                         missing, detailed):
     """
     Compute SMATCH on two files with pairwise AMRs, one-AMR-per-line. 
     """
     ps, rs, fs = [], [],[]
     try:
        gold_file = open(gold_filename)
     except IOError:
        sys.stderr.write("ERROR: Could not open gold AMR file %s.\n" % gold_filename)        
        sys.exit(1)
     try:
        test_file  =open(test_filename)
     except IOError:
        sys.stderr.write("ERROR: Could not open test AMR file %s.\n" % test_filename)        
        sys.exit(1)

     tiburonfailct = 0
     parsefailct = 0
     totalct = 0
     decodefailct = 0
     emptylinect = 0

     while True: 
            gold = gold_file.readline()
            test = test_file.readline().strip()
            if not gold: # EOF 
                break        
            gold = gold.strip()
            if not gold: 
                sys.stderr.write("WARNING: Empty line in gold AMR file. Skipping entry.\n")
                continue
            totalct += 1
            if gold:  
                try:
                    if concept_edges: # rebuild normal AMR with concepts attached to nodes.
                        amr_gold = Hgraph.from_string(gold)
                        amr_gold = Hgraph.from_concept_edge_labels(amr_gold) 
                    else:
                        amr_gold = Hgraph.from_string(gold)
                    l = len(amr_gold.triples())    
                except Exception as e:     
                    print >>sys.stderr, e
                    sys.stderr.write("WARNING: Could not parse gold AMR. Skipping entry.\n")    
                    continue

                if test and not test.startswith("#"): 
                    try:
                        amr_test = Hgraph.from_string(test)
                        if concept_edges: # rebuild normal AMR with concepts attached to nodes.
                            amr_test = Hgraph.from_concept_edge_labels(amr_test)
                        else:
                            amr_test = Hgraph.from_string(test)
                        
                        if precise:
                            p,r,f = compute_smatch_precise(amr_gold, amr_test)
                        else:
                            p,r,f = compute_smatch_hill_climbing(amr_gold, amr_test,
                                                                 starts = starts, method = method,
                                                                 restart_threshold = restart_threshold)
                        if detailed: 
                            print "P:%f R:%f F:%f " % (p, r, f) 
                        else: 
                            sys.stdout.write(".")
                            sys.stdout.flush()
                        ps.append((p,l)) 
                        rs.append((r,l)) 
                        fs.append((f,l)) 

                    except pyparsing.ParseException:
                        parsefailct += 1
                else:
                    if not missing:
                        rs.append((0.0, l))        
                        ps.append((0.0, l))        
                        fs.append((0.0, l))        
            else: 
                if test=="# Tiburon failed.":
                    tiburonfailct += 1
                elif test=="# Decoding failed.":
                    decodefailct += 1
                emptylinect += 1                    
                if not missing:
                    rs.append((0.0, l))        
                    ps.append((0.0, l))        
                    fs.append((0.0, l))        
                    
     sys.stdout.write("\n")                    
     avgp = mean(ps) 
     avgr = mean(rs)
     avgf = mean(fs)
     print "Total: %i\tFail(empty line): %i\tFail(invalid AMR): %i"  % (totalct, emptylinect, parsefailct)
     print "MEAN SMATCH: P:%f R:%f F:%f " % (avgp, avgr, avgf)


#amr1 = Hgraph.from_string("""
#        (s / say-01 :ARG0 (p2 / professor :location (c2 / college :mod (m / 
#medicine) :poss (u2 / university :name (u3 / name :op1 "University" :op2 
#"of" :op3 "Vermont"))) :mod (p / pathology) :name (b / name :op1 
#"Brooke" :op2 "T." :op3 "Mossman")) :ARG1 (i2 / include-91 :ARG1 (c / 
#country :name (u / name :op1 "U.S.")) :ARG2 (n / nation :ARG0-of (h / 
#have-03 :ARG1 (s2 / standard :mod (h2 / high :degree (m2 / more) ) 
#:prep-with-of (r / regulate-01 :ARG1 (f2 / fiber :ARG0-of (r2 / 
#resemble-01 :ARG1 (n2 / needle) ) :ARG1-of (c4 / classify-01 :ARG2 (a / 
#amphobile) ) :mod (s3 / smooth) :prep-such-as (c3 / crocidolite) ))) 
#:polarity -) :ARG1-of (i3 / include-91 :ARG2 (n3 / nation :ARG1-of (i / 
#industrialize-01) ) :ARG3 (f / few) ))))""")

#amr2 = Hgraph.from_string("""
#        (v0 / say-01 :ARG0 (v1 / professor :location (v3 / college) :mod (v2 / 
#pathology :mod (v4 / medicine :op1 (v11 / "university") :op2 (v12 / 
#"of") :op3 (v13 / "vermont") ) :poss (v5 / university :name (v6 / name) 
#)) :name (v7 / name :op1 (v8 / "brooke") :op2 (v9 / "t-period-") :op3 
#(v10 / "mossman") )) :ARG1 (v14 / include-91 :ARG1 (v15 / country :name 
#(v16 / name :op1 (v17 / "u-period-s-period-") )) :ARG2 (v18 / nation 
#:ARG0-of (v19 / have-03 :ARG1 (v20 / standard :mod (v26 / high :degree 
#(v27 / more :ARG0-of (v30 / resemble-01 :ARG1 (v31 / needle) ) :ARG1-of 
#(v32 / classify-01 :ARG2 (v33 / amphobile) ) :mod (v34 / smooth) 
#:prep-such-as (v35 / crocidolite) )) :prep-with-of (v28 / regulate-01 
#:ARG1 (v29 / fiber) )) :polarity (v21 / -) ) :ARG1-of (v22 / include-91 
#:ARG2 (v23 / nation :ARG1-of (v24 / industrialize-01) ) :ARG3 (v25 / 
#few) ))))""")

def main():

    usage = "usage: %prog [options] gold_amr_file test_amr_file\nBoth AMR files are one-amr-per-line.\n Try %prog --help for more options."
    parser = OptionParser(usage = usage)

    parser.add_option("-m", "--method", dest="method", default="root",
                      help="Set method to initialize the variable alignment in each hillclimbing restart.\n 'random': Randomly map variables.\n'concept': Random mapping between variables with the same concept\n'root'(default): Random mapping, roots always mapped.\n'optimistic': Assume AMRs have the same structure and map variables. Do not restart.", metavar="METHOD")
    parser.add_option("-d", "--detailed",
                      action="store_true", dest="detailed", default=False,
                     help="Print scores per sentence.")
    parser.add_option("-r", "--restarts",
                      action="store", dest="restarts", default="10", metavar="RESTARTS",
                      help="Restart hill climbing RESTARTS time. 1 means single start. Default: 10.")
    parser.add_option("-t", "--threshold",
                      action="store", dest="threshold", default=1.0, metavar="VALUE",
                      help="Do not restart hill climbing after F-Score reaches VALUE in [0.0-1.0]. Default: 1.0")
    parser.add_option("-c", "--concept_edges",
                      action="store_true", dest="concept_edges", default=False, 
                      help="Do not restart hill climbing after F-Score reaches this threshold [0.0-1.0]")
    parser.add_option("-i", "--ignore-missing",
                      action="store_true", dest="missing", default=False,
                      help="Ignore line pairs with empty test AMR when computing overall Smatch.")

    group = OptionGroup(parser, "Dangerous Options",
                       "Caution: These options are believed to bite! Use at your own risk.")
    group.add_option("-p", "--precise",
                      action="store_true", dest="precise", default=False,
                      help="Compute precise SMATCH score. Do not use this option!")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()
    if len(args)!=2:
        print parser.get_usage()
        sys.exit(1)
    try:
        restarts =  int(options.restarts) 
        if restarts <= 0:
            raise ValueError
    except: 
        sys.stderr.write("ERROR: Invalid number of restarts. Must be an integer > 0.\n")
        print parser.get_usage()
        sys.exit(1)
    try:
        threshold = float(options.threshold)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError/tmp/gold.amr
    except:
        sys.stderr.write("ERROR: Invalid restart threshold. Must be a float 0.0<=i<=1.0\n")
        print parser.get_usage()
        sys.exit(1)

    concept_edges = options.concept_edges
    missing = options.missing
    precise = options.precise
    detailed = options.detailed

    if precise:
        sys.stderr.write("WARNING: Computing precise SMATCH (i.e. graph isomorphism!). This can take very long!\n")
    
    method = options.method
    method_map = {'random':get_random_start,
                  'concept':get_concept_match_start,
                  'root':get_root_align_start,
                  'optimistic':get_parallel_start
     }
    if not method in method_map:
        sys.stderr.write("ERROR: -m METHOD must be one of the following: random, concept, root, optimistic.\n")        
        print parser.get_usage()
        sys.exit(1)

    if method == "optimistic": 
        print "Using optimistic initializer. Setting number of restarts to 1."
        restarts = 1


    compute_smatch_batch(args[0], args[1], starts = restarts, method = get_random_start,                         
                         restart_threshold = threshold, concept_edges= concept_edges, precise = precise, missing = missing, detailed = detailed)

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    
    #print "go"   
    #t = timeit.Timer("print compute_smatch_hill_climbing(amr1,amr2,1)","from __main__ import compute_smatch_hill_climbing,amr1, amr2")
    #
    #print "%f sec" % t.timeit(number=1)
 
    main()
