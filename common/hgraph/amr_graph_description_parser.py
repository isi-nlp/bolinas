from collections import defaultdict
import unittest
import re
import sys
from hgraph import Hgraph, SpecialValue, StrLiteral, Quantity, Literal, NonterminalLabel, Polarity

"""
A deterministic, linear time parser for Penman-style graph/meaning 
representation descriptions. 
The graph format is described here...

"""

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
        Perform the lexical scanning on a string and yield a (type, token, position)
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
            type = match.lastgroup
            yield type, token, position

class LexTypes:
    """
    Definitions of lexical types returned by the lexer.
    """
    LPAR = "LPAR" 
    RPAR = "RPAR"
    COMMA = "COMMA" 
    SLASH = "SLASH" 
    EDGELABEL = "EDGELABEL" 
    STRLITERAL = "STRLITERAL" 
    IDENTIFIER = "IDENTIFIER" 
    LITERAL =  "LITERAL"
    QUANTITY = "QUANTITY"
    POLARITY = "POLARITY"

# Parser
class GraphDescriptionParser(object):
    """
    A deterministic, linear time parser for Penman-style graph descriptions. 

    >>> parser = GraphDescriptionParser()
    >>> x = parser.parse_string('''(x :foo (a /c1 :bar (b / c2 :#FOO (d :blubb "HI") ,@c :value 10))) (y :foo (d :size 'small))''')
    >>> x
    DAG{ (x :foo (a / c1 :bar (b / c2 :#FOO @c ,(d :blubb "HI" :size 'small) :value 10))) (y :foo d) }
    >>> x.external_nodes
    ['c']
    >>> x.roots
    ['x', 'y']
    >>> parser.parse_string('''(x :foo (a /c1 :bar (b / c2 :#FOO (d :blubb "HI") ,@c :value 10))) (y :foo (d :size 'small))''', concepts = False)
    DAG{ (x :foo (a :bar (b :#FOO @c, (d :blubb "HI" :size 'small) :value 10))) (y :foo d) }
    """
    def __init__(self):
        # Lexical 
        lex_rules = [
            (LexTypes.LPAR, '\('),
            (LexTypes.RPAR,'\)'),
            (LexTypes.COMMA,','), 
            (LexTypes.SLASH,'/'),
            (LexTypes.EDGELABEL,":[^\s\)]+"),
            (LexTypes.STRLITERAL,'"[^"]+"(?:~[0-9]+)?'),
            (LexTypes.LITERAL,"'[^\s(),]+(?:~[0-9]+)?"),
            (LexTypes.POLARITY,"-"),
            (LexTypes.QUANTITY,"[0-9][0-9Ee^+\-\.,:]*(?:~[0-9]+)?"),
            (LexTypes.IDENTIFIER,"[^\s(),]*")
        ] 
        self.lexer = Lexer(lex_rules)

    def parse_string(self, s, concepts = True):
        """
        Parse the string s and return a new abstract meaning representation.

        @concepts if True, method returns an L{Hgraph} object containing concept labels. 
        """

        PNODE = 1
        CNODE = 2
        EDGE = 3

        amr = Hgraph()
        stack = [] 
        state = 0

        new_node_counter = 0

        #0, top leve
        #1, expecting source nodename
        #2, expecting concept name or edge label
        #3, lexpecting concept name 
        #4, expecting edge label
        #5, expecting expression, node name or literal string, quantity or special symbol   
        #6, expecting right paren or more target nodes
        #7, expecting right paren

        for type, token, pos in self.lexer.lex(s):

            if state == 0:
                if type == LexTypes.LPAR:
                    state = 1
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 1:
                if type == LexTypes.IDENTIFIER:
                    alignments = token.split("~")
                    nodelabel = alignments[0]
                    stack.append((PNODE, nodelabel, None)) # Push source node
                    for alignment in alignments[1:]: 
                        amr.node_alignments[nodelabel] = (int(alignment),)
                    state = 2
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 2:
                if type == LexTypes.SLASH:
                    state = 3
                elif type == LexTypes.EDGELABEL:
                    stack.append((EDGE, token[1:]))
                    state = 5
                elif type == LexTypes.RPAR:
                    forgetme, parentnodelabel, parentconcept = stack.pop()
                    assert forgetme == PNODE
                    if parentnodelabel[0] == '@': 
                        parentnodelabel = parentnodelabel[1:]
                        amr.external_nodes.append(parentnodelabel)
                    foo =  amr[parentnodelabel] # add only the node
                    if stack:
                        stack.append((CNODE, parentnodelabel, parentconcept))
                        state = 6
                    else:    
                        amr.roots.append(parentnodelabel)
                        state = 0

                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 3:
                if type == LexTypes.IDENTIFIER:
                    assert stack[-1][0] == PNODE
                    nodelabel = stack.pop()[1]
                    stack.append((PNODE, nodelabel, token)) # Push new source node with concept label
                    state = 4
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 4:
                if type == LexTypes.EDGELABEL:
                    stack.append((EDGE, token[1:]))
                    state = 5
                elif type == LexTypes.RPAR:
                    forgetme, parentnodelabel, parentconcept = stack.pop()
                    assert forgetme == PNODE
                    if parentnodelabel[0] == '@': 
                        parentnodelabel = parentnodelabel[1:]
                        amr.external_nodes.append(parentnodelabel)
                    foo = amr[parentnodelabel] # add only the node
                    if concepts and (not parentnodelabel in amr.node_to_concepts or parentnodelabel is not None): 
                        amr.node_to_concepts[parentnodelabel] = parentconcept    
                    if stack: 
                        stack.append((CNODE, parentnodelabel, parentconcept))
                        state = 6
                    else:    
                        amr.roots.append(parentnodelabel)
                        state = 0
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 5:
                if type == LexTypes.LPAR:
                    state = 1
                elif type == LexTypes.POLARITY:
                    alignments = token.split("~")
                    polarity = alignments[0] 
                    node_id = "_%i" % new_node_counter
                    new_node_counter += 1
                    for alignment in alignments[1:]:
                        amr.node_alignments[node_id] = (int(alignment),)
                    #stack.append((CNODE, Quantity(quantity), None))
                    stack.append((CNODE, node_id, Polarity(polarity)))
                    state = 6
                elif type == LexTypes.QUANTITY:
                    alignments = token.split("~")
                    quantity= alignments[0]    
                    node_id = "_%i" % new_node_counter
                    new_node_counter += 1
                    for alignment in alignments[1:]:
                        amr.node_alignments[node_id] = (int(alignment),)
                    #stack.append((CNODE, Quantity(quantity), None))
                    stack.append((CNODE, node_id, Quantity(quantity)))
                    state = 6
                elif type == LexTypes.STRLITERAL:
                    alignments = token.split("~")
                    strliteral = alignments[0]    
                    node_id = "_%i" % new_node_counter
                    new_node_counter += 1
                    for alignment in alignments[1:]:
                        amr.node_alignments[node_id] = (int(alignment),)
                    #stack.append((CNODE, StrLiteral(strliteral[1:-1]), None))
                    stack.append((CNODE, node_id, StrLiteral(strliteral[1:-1])))
                    state = 6
                elif type == LexTypes.LITERAL:
                    alignments = token.split("~")
                    literal = alignments[0]    
                    node_id = "_%i" % new_node_counter
                    new_node_counter += 1
                    for alignment in alignments[1:]:
                        amr.node_alignments[node_id] = (int(alignment),)
                    #stack.append((CNODE, Literal(literal[1:]), None)) 
                    stack.append((CNODE, node_id , Literal(literal[1:]))) 
                    state = 6
                elif type == LexTypes.IDENTIFIER: 
                    alignments = token.split("~")
                    nodelabel = alignments[0]
                    stack.append((CNODE, nodelabel, None)) # Push new child node with concept label
                    for alignment in alignments[1:]:
                        amr.node_alignments[nodelabel] = (int(alignment),)
                    state = 6
                elif type == LexTypes.EDGELABEL:  # Unary edge
                    stack.append((CNODE, None, None))
                    stack.append((EDGE, token[1:]))
                    state = 5
                        
                elif type == LexTypes.RPAR: # Unary edge
                    stack.append((CNODE, None, None))             
                    edges = []
                    while stack[-1][0] != PNODE: # Pop all edges
                        children = []
                        while stack[-1][0] == CNODE: # Pop all external nodes for hyperedge
                            forgetme, childnodelabel, childconcept = stack.pop()
                            if childnodelabel is not None and childnodelabel[0] == '@': #child is external node
                                childnodelabel = childnodelabel[1:]
                                amr.external_nodes.append(childnodelabel)
                            children.append((childnodelabel, childconcept))

                        assert stack[-1][0] == EDGE 
                        forgetme, edgelabel = stack.pop()
                        edges.append((edgelabel, children))
                   
                    forgetme, parentnodelabel, parentconcept = stack.pop()
                    if concepts and (not parentnodelabel in amr.node_to_concepts or parentconcept is not None): 
                        amr.node_to_concepts[parentnodelabel] = parentconcept
                    if parentnodelabel[0] == '@': #parent is external node
                        parentnodelabel = parentnodelabel[1:]
                        amr.external_nodes.append(parentnodelabel)
                    for edgelabel, children in edges: 

                        hypertarget =[] # build hyperedge destination
                        for node, concept in children:
                            if node is not None:
                                if concepts and (not node in amr.node_to_concepts or concept is not None):
                                    amr.node_to_concepts[node] = concept
                                hypertarget.append(node) 
                        hyperchild = tuple(hypertarget)    
                        
                        if edgelabel[0] == '#': # this is a nonterminal Edge 
                            edgelabel = NonterminalLabel(edgelabel[1:])

                        amr._add_triple(parentnodelabel, edgelabel, hyperchild)

                    if stack:
                        state = 6
                        stack.append((CNODE, parentnodelabel, parentconcept))
                    else: 
                        state = 0 
                        amr.roots.append(parentnodelabel)
                     
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 6:
                if type == LexTypes.RPAR: # Pop from stack and add edges

                    edges = []
                    
                    while stack[-1][0] != PNODE: # Pop all edges
                        children = []
                        while stack[-1][0] == CNODE: # Pop all external nodes for hyperedge
                            forgetme, childnodelabel, childconcept = stack.pop()
                            if childnodelabel is not None and childnodelabel[0] == '@': #child is external node
                                childnodelabel = childnodelabel[1:]
                                amr.external_nodes.append(childnodelabel)
                            children.append((childnodelabel, childconcept))

                        assert stack[-1][0] == EDGE 
                        forgetme, edgelabel = stack.pop()
                        edges.append((edgelabel, children))
                   
                    forgetme, parentnodelabel, parentconcept = stack.pop()
                    if concepts and (not parentnodelabel in amr.node_to_concepts or parentconcept is not None): 
                        amr.node_to_concepts[parentnodelabel] = parentconcept
                    if parentnodelabel[0] == '@': #parent is external node
                        parentnodelabel = parentnodelabel[1:]
                        amr.external_nodes.append(parentnodelabel)
                    for edgelabel, children in edges: 

                        hypertarget =[] # build hyperedge destination
                        for node, concept in children:
                            if node is not None: 
                                if concepts and (not node in amr.node_to_concepts or concept is not None):
                                    amr.node_to_concepts[node] = concept
                                hypertarget.append(node) 
                        hyperchild = tuple(hypertarget)    
                        
                        if edgelabel[0] == '#': # this is a nonterminal Edge 
                            edgelabel = NonterminalLabel(edgelabel[1:])
                        amr._add_triple(parentnodelabel, edgelabel, hyperchild)

                    if stack:
                        state = 6
                        stack.append((CNODE, parentnodelabel, parentconcept))
                    else: 
                        state = 0 
                        amr.roots.append(parentnodelabel)
                        
                elif type == LexTypes.COMMA:
                    state = 7

                elif type == LexTypes.EDGELABEL: 
                    stack.append((EDGE, token[1:]))
                    state = 5

                else: raise ParserError, "Unexpected token %s at position %i of type %s." % (token, pos, type)

            elif state == 7: 
                if type == LexTypes.IDENTIFIER:
                    alignments = token.split("~")
                    nodelabel = alignments[0]
                    stack.append((CNODE, nodelabel, None)) # Push new target node with concept label
                    for alignment in alignments[1:]:
                        amr.node_alignments[nodelabel] = (int(alignment),)
                    state = 6
                elif type== LexTypes.LPAR:
                    state = 1
                else: raise ParserError, "Unexpected token %s at position %i of type %s." % (token, pos, type)

        return amr

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()


    #import timeit
    #
    #s = """for line in lines: 
    #         parser.parse_string(line)"""
    #t = timeit.Timer(stmt = s, setup = """from graph_description_parser import GraphDescriptionParser\nlines = open(sys.argv[1],'r').readlines()\nparser = GraphDescriptionParser()""")
    #print t.timeit(number = 1)
    #s2 = """for line in lines: 
    #         Hgraph.from_string(line)"""
    #t2 = timeit.Timer(stmt = s2, setup = """from amr import Hgraph\nlines = open(sys.argv[1],'r').readlines()""")
    #print t2.timeit(number = 1)

    parser = GraphDescriptionParser()
    with open(sys.argv[1]) as in_file:
        for line in in_file:
            print parser.parse_string(line).to_string(newline= False)

