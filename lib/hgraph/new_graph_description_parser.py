import sys
from collections import defaultdict

import re

from hgraph import Hgraph, SpecialValue, StrLiteral, Quantity, Literal
from lib.cfg import NonterminalLabel

"""
A deterministic, linear time parser for hypergraph descriptions.
The graph format is described here...

@author Daniel Bauer (bauer@cs.columbia.edu)
@date 2013-06-10
"""

# This is essentially a finite state machine with a single stack and 
# semantic actions on each transition.
# Input symbols are provided by a lexer. 
# This hand-written implementation is MUCH faster than pyparsing or 
# automatically generated parsers. 

# Error definitions
class LexerError(Exception):
    pass
class ParserError(Exception):
    pass


# Lexer
class Lexer(object):
    """
    A simple generic lexer using Python re, that accepts a list of token
    definitions and ignores whitespaces.
    """
    def __init__(self, rules):
        """
        Initialize a new Lexer object using a set of lexical rules. 

        @type rules: A list of tuples (lextype, regex) where lextype is a 
        string identifying the lexical type of the token and regex is a python
        regular expression string. The order of tuples in the list matters.
        """
        self.tokenre = self.make_compiled_regex(rules)
        self.whitespacere = re.compile('[\s]*', re.MULTILINE)

    def make_compiled_regex(self, rules):
        regexstr =  '|'.join('(?P<%s>%s)' % (name, rule) for name, rule in rules)
        return re.compile(regexstr)

    def lex(self, s):
        """
        Perform lexical scanning on a string and yield a (type, token, position)
        triple at a time. Whitespaces are skipped automatically.  
        This is a generator, so lexing is performed lazily. 
        """
        position = 0
        s = s.strip()
        while position < len(s):

            # Skip white spaces
            match = self.whitespacere.match(s, position)
            if match: 
                position = match.end()
    
            match = self.tokenre.match(s, position)
            if not match:
                raise LexerError, "Could not tokenize '%s'" % re.escape(s[position:])
            position = match.end()
            token = match.group(match.lastgroup)
            typ = match.lastgroup
            yield typ, token, position


class LexTypes:
    """
    Definitions of lexical types returned by the lexer.
    """
    LPAR = "LPAR" 
    RPAR = "RPAR"
    COMMA = "COMMA" 
    NODE= "NODE" 
    EQUALS= "EQUALS" 
    EDGELABEL = "EDGELABEL" 
    STRLITERAL = "STRLITERAL" 
    IDENTIFIER = "IDENTIFIER" 
    LITERAL =  "LITERAL"
    QUANTITY = "QUANTITY"


# Parser
class GraphDescriptionParser(object):
    """
    A deterministic, linear time parser for hypergraph descriptions. 
    See documentation for hypergraph format.

    >>> parser = GraphDescriptionParser()
    """
    def __init__(self):
        # Lexical 
        lex_rules = [
            (LexTypes.LPAR, '\('),
            (LexTypes.RPAR,'\)'),
            (LexTypes.EDGELABEL,':[^\s\)]*'),
            (LexTypes.NODE,'[^\s(),.]*\.?[^\s(),.]*')
        ] 
        self.node_re = re.compile('([^\s(),.*]*)(\.?)([^\s(),.*]*)(\*?([0-9]*))')
        self.lexer = Lexer(lex_rules)

        self.id_count = 0
        self.nt_id_count = 0
        self.ext_id_count = 0
        self.explicit_ext_ids = False

    def parse_node(self, token):
        """
        Parse an individual node of the format [id].[label][*[id]] or just a label.
        """
        match = self.node_re.match(token)
        groups = match.groups()
        
        
        if not groups[1]:    #This node is only a label
            label = groups[0]
            ident = "_%i" % self.id_count
            self.id_count += 1       
        else:               
            label = groups[2]
            if not groups[0]: # Found a . but no explicit id
                ident = "_%i" % self.id_count
                self.id_count += 1       
            else: 
                ident = groups[0] 
        
        if groups[3]:       #Check if node is an external node
            if groups[4]:   #Get external node ID
                ext_id = int(groups[4]) 
                if not self.explicit_ext_ids and self.ext_id_count >= 1:
                    raise LexerError, "Must specify explicit external node IDs for all or none of the external nodes."
                self.explicit_ext_ids = True
            else:
                if self.explicit_ext_ids:
                    raise LexerError, "Must specify explicit external node IDs for all or none of the external nodes."
                if not ident in self.seen_nodes: #UGLY
                    self.seen_nodes.add(ident)
                    ext_id = self.ext_id_count
                    self.ext_id_count += 1
        else:
            ext_id = None

        return ident, label, ext_id 

    def parse_string(self, s, concepts = True):
        """
        Parse the string s and return a new hypergraph. 
        """

        # Constants to identify items on the stack
        PNODE = 1 # Parent node
        CNODE = 2 # Child node
        EDGE = 3  # Hyperedge 

        amr = Hgraph()
        
        stack = []
        state = 0

        self.id_count = 0
        self.nt_id_count = 0
        self.ext_id_count = 0
        self.seen_nodes = set()
        self.explicit_ext_ids = False                 
 
        # States of the finite state parser
        #0, top level
        #1, expecting head nodename
        #2, expecting edge label or node
        #3, expecting further child nodes or right paren
        #4, expecting saw edge label, expecting child node, edge label, right paren 

        def insert_node(node, root=False):
            # Insert a node into the AMR
            ident, label, ext_id = node                              
            ignoreme = amr[ident] #Initialize dictionary for this node
            amr.node_to_concepts[ident] = label
            if ext_id is not None:                
                if ident in amr.external_nodes and amr.external_nodes[ident] != ext_id:
                    raise ParserError, "Incompatible external node IDs for node %s." % ident
                amr.external_nodes[ident] = ext_id
                amr.rev_external_nodes[ext_id] = ident
            if root: 
                amr.roots.append(ident)
                
        def pop_and_transition():
            # Create all edges in a group from the stack, attach them to the 
            # graph and then transition to the appropriate state in the FSA
            edges = []
            while stack[-1][0] != PNODE: # Pop all edges
                children = []
                while stack[-1][0] == CNODE: # Pop all nodes in hyperedge
                    itemtype, node = stack.pop()
                    insert_node(node) 
                    children.append(node)
                assert stack[-1][0] == EDGE 
                itemtype, edgelabel = stack.pop()
                edges.append((edgelabel, children))
              
            # Construct the hyperedge 
            itemtype, parentnode = stack.pop()
            for edgelabel, children in edges: 
                hypertarget = [] # build hyperedge tail 
                for ident, label, ext_id in children:
                    hypertarget.append(ident) 
                hypertarget.reverse()
                hyperchild = tuple(hypertarget)    
                
                if "$" in edgelabel: # this is a nonterminal Edge 
                    new_edge = NonterminalLabel.from_string(edgelabel)
                    if not new_edge.index:
                        new_edge.index = "_%i" %self.nt_id_count
                        self.nt_id_count = self.nt_id_count + 1
                else: 
                    new_edge = edgelabel
                ident, label, ext_id = parentnode
                amr._add_triple(ident, new_edge, hyperchild) 
               
            if stack:
                insert_node(parentnode)
                stack.append((CNODE, parentnode))
                state = 4
            else:    
                insert_node(parentnode, root = True)
                state = 5

        # Parser transitions start here
        for typ, token, pos in self.lexer.lex(s):

            if state == 0:
                if typ == LexTypes.LPAR:
                    state = 1
                elif typ == LexTypes.NODE:
                    insert_node(self.parse_node(token), root=True)               
                    state = 5
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)
             
            elif state == 1: 
                if typ == LexTypes.NODE:
                    stack.append((PNODE, self.parse_node(token))) # Push head node
                    state = 2
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 2:
                if typ == LexTypes.EDGELABEL:
                    stack.append((EDGE, token[1:]))
                    state = 4
                elif typ == LexTypes.NODE:
                    stack.append((EDGE, "")) # No edge specified, assume empty label
                    stack.append((CNODE, self.parse_node(token))) 
                    state = 3
                elif typ == LexTypes.LPAR:
                    stack.append((EDGE, "")) # No edge specified, assume empty label
                    state = 1
                elif typ == LexTypes.RPAR:
                    itemtype, node  = stack.pop()
                    assert itemtype == PNODE
                    if stack:
                        insert_node(node)
                        stack.append((CNODE, node))
                        state = 3
                    else:    
                        insert_node(node, root = True)
                        state = 5
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 3:
                if typ == LexTypes.RPAR: # Pop from stack and add edges
                    pop_and_transition(); 
                elif typ == LexTypes.NODE:
                    stack.append((CNODE, self.parse_node(token)))
                    state = 3
                elif typ == LexTypes.EDGELABEL:
                    stack.append((EDGE, token[1:]))
                    state = 4
                elif typ == LexTypes.LPAR:
                    state = 1
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 4:
                if typ == LexTypes.LPAR:
                    state = 1
                elif typ == LexTypes.NODE:
                    stack.append((CNODE, self.parse_node(token))) 
                    state = 3
                elif typ == LexTypes.EDGELABEL:
                    stack.append((EDGE, token[1:]))
                elif typ == LexTypes.RPAR: # Pop from stack and add edges
                    pop_and_transition(); 
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)
            
            elif state == 5:
                raise ParserError, "Unexpected token %s at position %i." % (token, pos)

        return amr

if __name__ == "__main__":
    # Just test the module
    import doctest
    doctest.testmod()

    parser = GraphDescriptionParser()
