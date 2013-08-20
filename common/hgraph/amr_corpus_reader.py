#!/usr/bin/env python2
from hgraph import Hgraph
import amr_graph_description_parser

#import tree
import re
import sys
import string
from collections import defaultdict as ddict

def format_tagged(s):
  #return [tuple(p.split('/')) for p in s.split()]
  return [p.rsplit('-',1)[0] for p in s.split()]

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
            sys.exit(1)

    amr_triple = (var, role, (filler,))    
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
        p = SentenceWithHgraph(sent_id, sent_id,  amr, tagged, None, alignments)   
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
        p = SentenceWithHgraph(sent_id, sent_no, amr, tagged, None, None)   
        yield p

def metadata_amr_corpus(f):
    """
    Read the next parsed sentence from an input file using the AMR meta data format.
    """
    metadata = []
    sentence = "" 
    sent_id = ""
    buff = []
    idmatcher = re.compile("# ::id ([^ ]+) ")
    sentmatcher = re.compile("# ::snt (.*)")
    
    count = 1

    parser = amr_graph_description_parser.GraphDescriptionParser()

    while True:        
        l = f.readline()
        if not l:
            raise StopIteration
          
        l = l.strip()     
        if not l:
            if buff: 
                amr = parser.parse_string(" ".join(buff))
                yield SentenceWithHgraph(sent_id, count, amr, sentence, metadata = metadata)
                count += 1

            buff = []
            metadata = []
            sentence = ""
            sent_id = ""
        
        elif l.startswith("#"):
            metadata.append(l)
            match = idmatcher.match(l)
            if match: 
                sent_id = match.group(1)
            match = sentmatcher.match(l)
            if match: 
                sentence = match.group(1)
        else: 
            buff.append(l)
        

class SentenceWithHgraph(): 
    """
    A data structure to hold Hgraph <-> sentence pairs with 
    PTB parses and token to Hgraph edge elignments.
    """
    def __init__(self, sent_id, sent_no, amr, tagged, ptb = None, edge_alignments = None, metadata = None):
        self.sent_no = sent_no
        self.sent_id = sent_id
        self.amr = amr
        self.tagged = tagged
        self.ptb = ptb 
        self.alignments = edge_alignments
        self.metadata = metadata



if __name__ == "__main__":
    in_f =  open(sys.argv[1],'r')
    a = metadata_amr_corpus(in_f)
