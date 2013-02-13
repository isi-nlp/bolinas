#!/usr/bin/env python2

import sys
import re
from lib.amr.amr import Amr
from collections import defaultdict as ddict
import subprocess

treewidths = []

while True:
  line = sys.stdin.readline()
  if not line:
    break

  tline = line.strip().lower()
  match = re.match(r'\d+\. (.*) \(\S+\)', tline)

  sys.stdin.readline()

  amr_parts = []
  while True:
    amr_line = sys.stdin.readline().strip()
    if amr_line == '':
      break
    amr_parts.append(amr_line)

  amr_string = ' '.join(amr_parts)
  amr = Amr.from_string(amr_string)

  edges = amr.triples(instances=False)
  node_names = {}
  for edge in edges:
    if edge[0] not in node_names:
      node_names[edge[0]] = len(node_names) + 1
    for n in edge[2]:
      if n not in node_names:
        node_names[n] = len(node_names) + 1

  with open('/tmp/graph', 'w') as gfile:
    print >>gfile, 'p cnf %d %d' % (len(node_names), len(edges))
    for edge in edges:
      print >>gfile, node_names[edge[0]],
      for n in edge[2]:
        print >>gfile, node_names[n],
      print >>gfile, '0'
    gfile.close()

    output = subprocess.check_output(['/home/jacob/quickbb_64', '--cnffile',
      '/tmp/graph']).split('\n')
    tw = int(output[-2].split()[1])
    treewidths.append(tw)

print 'average', sum(treewidths) / (1.0 * len(treewidths))
print 'min', min(treewidths)
print 'max', max(treewidths)
print 'median', treewidths[len(treewidths)/2]
