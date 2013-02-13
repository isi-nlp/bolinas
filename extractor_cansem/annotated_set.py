from lib.amr.dag import Dag

from data_structures import Edge

def loadData(nl_path,mr_path,alignment_path):
    """
    Loads input data (nl,mr,alignment)
    """

    nl_f = open(nl_path)
    mr_f = open(mr_path)
    al_f = open(alignment_path)

    
    n_examples = count_lines(nl_path)

    # load input data into examples list
    data = []
    for i in range(n_examples):
        nl_s = nl_f.readline().strip()
        mr_s = mr_f.readline().strip()
        al_s = al_f.readline().strip()
        data.append({})

        data[i]['nl'] = nl_s
        data[i]['al'] = al_s

        # Stores a list of triples representing the MR aligned to natural language
        data[i]['mrt'] = applyAlign(loadMR(mr_s),al_s)

    return data


def loadMRFile(mr_path):
    """
    Loads input data (nl,mr,alignment)
    """

    mr_f = open(mr_path)
    n_examples = count_lines(mr_path)

    # load input data into examples list
    data = []
    for i in range(n_examples):
        mr_s = mr_f.readline().strip()
        data.append({})

        # Stores a list of triples representing the MR aligned to natural language
        data[i]['mrt'] = loadMR(mr_s)

    return data


def count_lines(filename):
  """
  Counts the number of lines in the given file.
  """
  n_lines = 0
  with open(filename) as f:
    for line in f:
      n_lines += 1
  return n_lines


def loadMR(mr):
    """
    Load meaning representation using Dag class.
    Store only the set of triples, which will be required throughout
    """
    dag = Dag.from_string(mr) #.stringify()
    triples = []

    for triple in dag.triples(instances=False):
        triples.append((triple[0],Edge(triple[1],1,1),triple[2][0]))
    #amr = Dag.from_triples(triples)
    return triples

def applyAlign(mrt,al):
    """
    Takes meaning representation triples (mrt) and combines with alignments
    """

    for alignment in al.split():
        # Alignment: x9:arg0:sell:x11-39
        fromNode,rest       = alignment.split(":",1)
        rest,toAlign        = rest.rsplit("-",1)
        edgeLabel,toNode    = rest.rsplit(":",1)
        hasAligned = False
        for i in xrange(len(mrt)):
            if mrt[i][0] == fromNode and mrt[i][2] == toNode and mrt[i][1][0] == edgeLabel: 
                mrt[i][1].align.add(toAlign)
                hasAligned = True
        if not hasAligned:
            print "Alignment failure in sentence #%d: (%s,%s,%s,%s)"%(i,fromNode,edgeLabel,toNode,toAlign)
    return mrt
