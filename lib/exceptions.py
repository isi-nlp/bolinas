class InvocationException(Exception):
    pass

class InputFormatException(Exception):
    pass

class BinarizationException(Exception):
    pass


class DerivationException(Exception):
    pass

# Graph Parser
class LexerError(Exception):
    pass
class ParserError(Exception):
    pass

# Grammar
class GrammarError(Exception):
    pass
