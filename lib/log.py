import sys
from lib.termcolor import colored

def pe(parts, color=None):
  print >>sys.stderr, colored(' '.join([str(s) for s in parts]), color)

def debug(*message):
  if debug in LOG:
    pe(message, 'green')

def chatter(*message):
  if chatter in LOG:
    pe(message)

def info(*message):
  if info in LOG:
    pe(message, 'green')

def warn(*message):
  if warn in LOG:
    pe(message, 'yellow')

def err(*message):
  if err in LOG:
    pe(message, 'red')

LOG = {info, err, warn, chatter}
