#!/usr/bin/env python2
from amr import Amr
from dag import Dag 

import tree
import re
import sys
import string
from collections import defaultdict as ddict

def format_tagged(s):
  #return [tuple(p.split('/')) for p in s.split()]
  return [p.rsplit('-',1)[0] for p in s.split()]

def format_amr(l):
  amr_s = ' '.join(l)
  amr_g = Amr.from_string(amr_s)
  return amr_g

def read_to_empty(f):
  lines = []
  while True:
    l = f.readline().strip()
    if not l: return lines
    lines.append(l)

def format_constituents(l):
  return nltk.tree.ParentedTree("\n".join(l))

def format_alignments(l, amr):
  """
  Parse alignment descriptions from file
  """
  r = [] 
  for a in l:
    m = re.match(r'(\S+)\s+:(\S+)\s+(\S+)\s+(.+)\-(\d+)', a)
    if m: 
        var = m.group(1)
        role = m.group(2)
        filler = m.group(3).replace('"','')
        token = m.group(4)
        token_id = int(m.group(5)) - 1 
    else: 
        m = re.match(r'ROOT\s+([^\-]+)\-(\d+)', a)
        if m:

            var = None
            role = "ROOT"
            filler = amr.roots[0].replace('"','')
            token = m.group(1)
            token_id = int(m.group(2)) - 1
        else:
            print a
            sys.exit(1)

    amr_triple = (var, role, (filler,))    
    #if var!="ROOT" and not amr.has_triple(*amr_triple):
    #        sys.stderr.write("WARNING: found alignment for %s, which is not a an AMR edge.\n" % str(amr_triple))
    r.append((amr_triple, token_id))

  return r

textlinematcher = re.compile("^(\d+)\.(.*?)\((.*)\)?$")

def format_text(l):
    match = textlinematcher.match(l.strip())
    if not match:
        raise ValueError, "Not a valid text line in Ulf corpus: \n %s \n"%l
    s_no = int(match.group(1))
    text = match.group(2).strip().split(" ")
    s_id = match.group(3).strip()
    return s_id, s_no,  text


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
        p = SentenceWithAmr(sent_id, sent_id,  amr, tagged, None, alignments)   
        yield p


def ulf_corpus(f):
    """
    Read the next parsed sentence from an input file using Ulf's format.
    """
    while True:
        l = f.readline()
        if not l:
            raise StopIteration        

        while l.strip().startswith("#") or not l.strip():
            l = f.readline()
            if not l:
                raise IOError, "AMR data file ended unexpectedly- sentence without AMR."
   
        sent_id, sent_no, tagged = format_text(l.strip())
        l = f.readline()   
        amr = format_amr(read_to_empty(f)) 
        p = SentenceWithAmr(sent_id, sent_no, amr, tagged, None, None)   
        yield p

class SentenceWithAmr(): 
    """
    A data structure to hold Amr <-> sentence pairs with 
    PTB parses and token to Amr edge elignments.
    """
    def __init__(self, sent_id, sent_no, amr, tagged, ptb, edge_alignments):
        self.sent_no = sent_no
        self.sent_id = sent_id
        self.amr = amr
        self.tagged = tagged
        self.ptb = ptb 
        self.alignments = edge_alignments



if __name__ == "__main__":
    with open(sys.argv[1],'r') as in_f:
        for amr in plain_corpus(in_f):
            print amr.to_string(newline = False)

            
