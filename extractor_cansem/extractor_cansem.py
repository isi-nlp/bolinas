import argparse
import sys

from file_structure import fileNames
from annotated_set import AnnotatedSet
from pipeline import executeTraining, executeStr2AMR, executeAMR2Str, executeStr2StrFiltered

class ExtractorCanSem:

  def __init__(self):
    pass

  @classmethod
  def help(self):
      return ExtractorCanSem.main(ExtractorCanSem(),"--help")

  def main(self, *args):

    parser = argparse.ArgumentParser(description='Canonical Semantic Extraction Algorithm for Synchronous Hyperedge Replacement Grammars (SHERG)',
                                     fromfile_prefix_chars='@',
                                     prog='%s extract-cansem'%sys.argv[0])
    parser.add_argument('action', metavar='action', type=str, help="Action: (t)rain|(a)mr2str|(s)tr2amr|(f)ull str2amr2str")
    parser.add_argument('dataset', type=str, help="Name of the dataset (e.g. geoquery)")
    parser.add_argument('experiment', type=str, help='Name for the current experiment')

    parser.add_argument('--experiment2', type=str, help='Name of the second experiment in case we use two separate things for translation')

    parser.add_argument('--language', '-l', dest='lang', type=str, nargs='+', help='Language of Input file (en,zh,fr,..)', default='en')
    parser.add_argument('--iterations', '-i', dest='iterations', type=int, help='Number of iterations for transducer training', default=50)
    parser.add_argument('-j', '--java', dest="javaGHKM", action='store_true')

    parser.add_argument('--from', dest='fromLine', type=int, help='Perform action from line', default=0)
    parser.add_argument('--to', dest='toLine', type=int, help='Perform action to line', default=None)
    parser.add_argument('--cppghkm_U', type=str, help='-U value for GHKM CPP', default="10")
    parser.add_argument('--cppghkm_l', type=str, help='-l value for GHKM CPP', default="1000:3")
    #parser.add_argument('--cppghkm_m', type=str, help='-m value for GHKM CPP', default="5000")

    parser.add_argument('--tib_sentenceK', type=int, help='-k value for Tiburon Tree->Str', default="50")
    
    parser.add_argument('--lmweight', type=float, help='Weight for LM prob as opposed to transducer prob', default="0.5")

    parser.add_argument('-n', '--nplist', dest="applyNPList", action='store_true', help="Apply the NP list (in extra/wordlist.txt)")
    parser.add_argument('-f', '--filter', dest="xrsfilter", action='store_true', help="Use XRS database and rule filtering")

    parser.add_argument('--align', type=str, help='Name of the Alignment strategy file part (e.g. dg for out/some.en.dg.align)', default="dg")
    parser.add_argument('--dataSub', type=str, help='Subfolder within the datastructure (train/test/dev)')

    parser.add_argument('--cross', type=int, help='Cross Validation (0 = off, 1-10 = which split', default=0)
    parser.add_argument('--tibStepSize', type=int, help='Tiburon Step Size (how many sentences per run)', default=20)

    parser.add_argument('--cppghkmLoc', nargs='?', default='/home/kmh/Files/Tools/ghkm/xrs-extract/bin/extract', help="Path to GHKM CPP executable")
    parser.add_argument('--dataRoot', nargs='?', default='/nfs/nlg/semmt/data/science', help="Path to data directory (parent of actual data directory)")
    parser.add_argument('--datasetRoot', nargs='?', default=False, help="Actual data directory name. Defaults to dataset name")
    parser.add_argument('--experimentRoot', nargs='?', default='/nfs/nlg/semmt/data/science/experiments', help="Path to experiments directory (parent of actual data directory)")
    parser.add_argument('--ghkmDir', nargs='?', default='/nfs/nlg/semmt/tools/stanford-ghkm-2010-03-08/', help="Path to GHKM jars (ghkm-revised.jar)")
    parser.add_argument('--tiburonLoc', nargs='?', default='/nfs/nlg/semmt/tools/tiburon-1.0/tiburon', help="Path to Tiburon executable file")
    parser.add_argument('--bleuScorerLoc', nargs='?', default='/nfs/nlg/semmt/tools/bleu_scorer/multi-bleu.perl', help="Path to Bleu scoring file")
    parser.add_argument('--srilmLoc', nargs='?', default='/usr/share/srilm/bin/i686-ubuntu/ngram-count', help="Path to SRILM ngram-count file")

    parser.add_argument('--xrsLoc', nargs='?', default='/home/nlg-02/pust', help="Root directory for pipeline-0.9 and sstring/stree scripts")

    args = parser.parse_args(args=args)
    if len(args.lang) == 1:     args.lang.append(args.lang[0])

    if not args.experiment2:    args.experiment2 = args.experiment

    if not args.dataSub:
        if args.action == "t" or args.action == "train":    args.dataSub = "train"
        else:                                               args.dataSub = "test"

    # Define function:
    if args.action == "t" or args.action == "train":    args.function = "train"
    if args.action == "s" or args.action == "str2amr":  args.function = "s2a"
    if args.action == "a" or args.action == "amr2str":  args.function = "a2s"
    if args.action == "f" or args.action == "full":     args.function = "s2s"

    args = fileNames(args,args.lang[0],args.lang[1])

    if args.action in ["t","train"]:
        data = AnnotatedSet.loadTraining(args,args.lang[0])
        lang = args.lang[0]
    elif args.action in ["s","str2amr","a","amr2str"]:
        data = AnnotatedSet.loadTextAndAMR(args,args.lang[0])
        lang = args.lang[0]
    elif args.action in ["f","full"]:
        data1 = AnnotatedSet.loadText(args,args.lang[0])
        data2 = AnnotatedSet.loadText(args,args.lang[1])
        lang1 = args.lang[0]
        lang2 = args.lang[1]

    # Figure out what to do:
    
    if args.action == "t" or args.action == "train":    executeTraining(args,data)
    elif args.action == "s" or args.action == "str2amr":  executeStr2AMR(args,data,lang)
    elif args.action == "a" or args.action == "amr2str":  executeAMR2Str(args,data,lang)
    elif args.action == "f" or args.action == "full":     
        if args.xrsfilter:  executeStr2StrFiltered(args,data1,data2,lang1,lang2,False)
        else:               executeStr2StrFiltered(args,data1,data2,lang1,lang2,False)
    
