import re

def one_line(string):
  ol = ' '.join(string.split('\n'))
  return re.sub(r'\s+', ' ', ol)
