#!/usr/bin/env python2
from hgraph import Hgraph

import nltk.tree
import re
import sys
import string
from collections import defaultdict as ddict

def format_amr(l):
  amr_s = ' '.join(l)
  amr_g = Hgraph.from_string(amr_s)
  return amr_g

def read_to_empty(f):
  lines = []
  while True:
    l = f.readline().strip()
    if not l: return lines
    lines.append(l)

def plain_corpus(f):

    while True: 
        x = read_to_empty(f)
        if not x: 
            raise StopIteration  
        amr = format_amr(x)
        yield amr

def aligned_corpus(f):
    """
    Read the next parsed sentence from an input file using the aligned AMR/tagged string format.
    """
    while True:
        l = f.readline()
        if not l:
            raise StopIteration        

        while l.strip().startswith("#") or l.strip().startswith("==") or not l.strip():
            l = f.readline()
            if not l:
                raise IOError, "AMR data file ended unexpectedly."
   
        sent_id = int(l)
        l = f.readline()   
        amr = format_amr(read_to_empty(f)) 
        tagged = format_tagged(f.readline())
        l = f.readline()
        alignments = format_alignments(read_to_empty(f), amr)
        p = SentenceWithHgraph(sent_id, sent_id,  amr, tagged, None, alignments)   
        yield p

if __name__ == "__main__":
    if len(sys.argv)!=2:
        sys.stderr.write("Wrong number of arguments. Excepts exactly one input file and writes to stdout.\n")
        sys.exit(1)
    try:
        in_f = open(sys.argv[1],'r')
    except IOError:
        sys.stderr.write("Cannot open %s for reading.\n" % sys.argv[1])
        sys.exit(1)
    for amr in plain_corpus(in_f):
        print amr.to_string(newline = False)

            
