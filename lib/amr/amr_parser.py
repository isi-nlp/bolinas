'''
Parse a feature-structure-style text encoding of an AMR.

@author: Daniel Bauer (dbauer)
@since: 2012-06-18
'''

import lib.pyparsing
from lib.pyparsing import Literal,Word,CharsNotIn, OneOrMore, ZeroOrMore,Forward,nums,alphas, Optional, ParserElement 
from collections import defaultdict
import unittest
import re
import sys
import copy


class StrLiteral(str):
        
    def __str__(self):
        return '"%s"' % "".join(self)

    def __repr__(self):
            return "".join(self)

class SpecialValue(str):
        pass


def make_amr_parser():
    """
    Pyparsing parser for AMRs. This will return an abstract syntax tree that
    needs to be converted into an AMR using ast_to_amr.
    """
    def debug(s, loc, tok):
        if len(tok) > 1:
            flat = [tok[0]] + tok[1:]
        else: 
            flat =  tok
        return flat

    def parse_concept_expr(s, loc, tok):       
        node_name = tok[0]
        concept_name = None

        roles = []
        if len(tok) > 1:
           if type(tok[1]) is tuple:
                roles = tok[1:]
           else: 
              concept_name = tok[1]
              if len(tok) > 2:
                roles = tok[2:]
        return (node_name, concept_name, roles)
   
    ParserElement.enablePackrat() # Hopefully no bug in here...


    def parse_role(s,loc,tok):
        if len(tok) >= 2:
            r, ch = tok[0], []
            for v in tok[1:]:
                if isinstance(v, StrLiteral):
                # Parse the node alignment and move it to the edge
                    parts = v.replace(" ","").rsplit("~",1)   
                    if len(parts) >= 2:
                        v, align = parts
                        v = StrLiteral(v)
                        r = "%s~%s" % (r.strip(), align.strip())
                elif isinstance(v, SpecialValue):
                    parts = v.replace(" ","").rsplit("~",1)
                    if len(parts) >= 2:                            
                        v, align = parts
                        v = StrLiteral(v)
                        r = "%s~%s" % (r.strip(), align.strip())
                ch.append(v)              
            return  r, ch       
        else:
            return tok[0] 

    # Number are all mapped to the same node in the graph because of interning
    parse_quantity = lambda s, loc, tok: StrLiteral(" ".join(tok)) #float(tok[0]) if "." in tok[0] else int(tok[0]) 

    parse_string_literal = lambda s, loc, tok: StrLiteral(" ".join(tok)) 

    parse_special_value = lambda s, loc, tok: SpecialValue(" ".join(tok))

    lpar  = Literal( "(" ).suppress()
    rpar  = Literal( ")" ).suppress()

    quantity = Word(nums+".,").setParseAction(parse_quantity)

    node_name =  Word(alphas+nums+"""@-_.~$/<>%&!+\*?^`"'""") #Word(alphas+nums+"_@.")  

    lit_string = Literal('"').suppress() + CharsNotIn('"') + Literal('"').suppress() 
    concept_name = lit_string | Word(alphas+nums+"""-_.,`~$/<>%&!+\*?^"'""")     
    role_name = Word(alphas+nums+"""-_.,~$/<>%&!+\*:?^`"'""") | Literal("#").suppress()+Word(alphas+nums+"[]-$_").setParseAction(lambda s, loc, tok: NonterminalLabel(tok[0]))     


    special_attr = (Literal("-") | Literal("interrogative") | Literal("SHOULD") | Literal("MUST") | Literal("HAVE-TO")| Literal("WOULD") | Literal("CAN") |  Literal("DARE-TO")| Literal("BE-TO") | Literal("MAY") | Literal("GOING-TO") | Literal("MIGHT") | Literal("USED-TO")) + Optional(Literal("~")+Word(alphas+nums+"."))
    

    expr = Forward()
    value =  expr |\
             quantity.setParseAction(parse_quantity) |\
             special_attr.setParseAction(parse_special_value) | \
             node_name |\
             (lit_string + Optional(Literal("~")+Word(alphas+nums+"."))).setParseAction(parse_string_literal) 
             
  

    valuelist = Forward()
    valuelist << (value + Literal(",").suppress() + valuelist | value).setParseAction(debug)
    role = (Literal(":").suppress() + role_name + valuelist).setParseAction(parse_role)
 
    expr.setParseAction(parse_concept_expr) 
    expr << (lpar +  node_name + Optional(Literal("/").suppress() + concept_name) + ZeroOrMore(role) + rpar)
    
    return expr 

class NonterminalLabel(object):
    """
    There can be multiple nonterminal edges with the same symbol. Wrap the 
    edge into an object so two edges do not compare equal.
    Nonterminal edges carry a nonterminal symbol and an index that identifies
    it uniquely in a rule.
    """

    label_matcher = re.compile("(?P<label>.*?)(\[(?P<index>.*)\])?$")

    def __init__(self, label, index = None):            
        if index is not None:
            self.label = label
            self.index = index  
        else: 
            match = NonterminalLabel.label_matcher.match(label)
            self.index = match.group("index")
            self.label = match.group("label")

    def __eq__(self, other):
        try: 
            return self.label == other.label and self.index == other.index
        except AttributeError:
            return False     
    
    def __repr__(self):
        return "NT(%s)" % str(self)

    def __str__(self):
        if self.index is not None:
            return "#%s[%s]" % (str(self.label), str(self.index))
        else: 
            return "#%s" % str(self.label)

    def __hash__(self):
        return 83 * hash(self.label) + 17 * hash(self.index)

