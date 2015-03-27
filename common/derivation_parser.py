import sys
import re

from common.exceptions import ParserError
from common.grammar import DummyItem, Grammar
from common import output
from parser.vo_rule import VoRule 
import fileinput

'''
A deterministic, linear time parser for hypergraph descriptions.
The hypergraph format is described in doc/hgraph_format.txt

@author Daniel Bauer (bauer@cs.columbia.edu)
@date 2013-06-10
'''

# This is essentially a finite state machine with a single stack and 
# semantic actions on each transition.
# Input symbols are provided by a lexer. 
# This hand-written implementation is MUCH faster than pyparsing or 
# automatically generated parsers. 

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


class DerivationLexTypes:
    """
    Definitions of lexical types returned by the lexer.
    """
    LPAR = "LPAR" 
    RPAR = "RPAR"
    IDENT = "IDENT" 

# Parser
class DerivationParser:
    """
    A deterministic, linear time parser for hypergraph descriptions. 
    See documentation for hypergraph format.
    """
    def __init__(self, grammar):
        # Lexical 
        lex_rules = [
            (DerivationLexTypes.LPAR, '\('),
            (DerivationLexTypes.RPAR,'\)'),
            (DerivationLexTypes.IDENT,'[^()\s]+')
        ] 
        self.lexer = Lexer(lex_rules)
        self.grammar = grammar


    def parse_string(self, s):
        """
        Parse the derivation. 
        """

        RULE = 1 # Parent node
        NT = 2 # Child node

        derivation = []        

        stack = []
        state = 0

        # Parser transitions start here
        for typ, token, pos in self.lexer.lex(s):
            if state == 0: # Rule
                if typ == DerivationLexTypes.IDENT:
                    rule = self.grammar[int(token)]
                    stack.append(DummyItem(rule))
                    #stack.append(token)
                    state = 1
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)
             
            elif state == 1:  
                if typ == DerivationLexTypes.LPAR:
                    state = 2
                elif typ == DerivationLexTypes.RPAR:
                    subtree = stack.pop()
                    if not stack:
                        return subtree
                    else:
                        nt = stack.pop()
                        stack.append((nt, subtree))
                    state = 2
                    
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)

            elif state == 2: # Nonterminal
                if typ == DerivationLexTypes.IDENT:
                    stack.append(tuple(token.split("$")))
                    state = 3
                elif typ == DerivationLexTypes.RPAR:
                    children = []
                    while type(stack[-1]) is tuple:
                        children.append(stack.pop())
                    parent = stack.pop()
                    subtree = ((parent,dict(children)))
                    stack.append(subtree)
                    state = 1
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)
            
            elif state == 3: 
                if typ == DerivationLexTypes.LPAR:
                    state = 0
                elif typ == DerivationLexTypes.RPAR:
                    children = []
                    while stack and type(stack[-1]) is tuple:
                        children.append(stack.pop())
                    parent = stack.pop()
                    subtree = ((parent,dict(children)))
                    stack.append(subtree)
                    state = 2
                else: raise ParserError, "Unexpected token %s at position %i." % (token, pos)
        assert len(stack)==1 
        return stack[0]

if __name__ == "__main__":
    grammar = Grammar.load_from_file(open(sys.argv[1],'r'), VoRule)
    parser = DerivationParser(grammar)
   
    for line in fileinput.input("-"):
        line = line.split("#")[0].strip()
        derivation = parser.parse_string(line.strip())
        print " ".join(output.apply_string_derivation(derivation))

    

