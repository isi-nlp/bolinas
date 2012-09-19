from parser.rule import Rule
from lib.exceptions import InvocationException, InputFormatException

MODE_STRING = 0
MODE_TREE = 1

class Binarizer:

  def __init__(self):
    pass

  @classmethod
  def help(self):
    """
    Returns the Binarizer help message.
    """
    return 'Usage: bolinas binarize <input_prefix> <output_prefix> ' + \
        '[mode (default string)]'

  def main(self, *args):
    if len(args) == 2:
      input_prefix, output_prefix = args
      mode = MODE_STRING
    elif len(args) == 3:
      input_prefix, output_prefix, mode_s = args
      if mode_s == 'string':
        mode = MODE_STRING
      elif mode_s == 'tree':
        mode = MODE_TREE
      else:
        raise InputFormatException("mode must be string or tree")
    else:
      raise InvocationException()

    grammar = Rule.load_from_file(input_prefix)
    binarized_grammar = {}

    next_rule_id = 0
    for rule_id in grammar:
      rule = grammar[rule_id]
      if mode == MODE_STRING:
        b_rules, next_rule_id = rule.binarize(next_rule_id)
      else: # mode = MODE_TREE
        b_rules, next_rule_id = rule.binarize_tree(next_rule_id)
      if not b_rules:
        continue
      for b_rule in b_rules:
        binarized_grammar[b_rule.rule_id] = b_rule

    Rule.write_to_file(binarized_grammar, output_prefix)
