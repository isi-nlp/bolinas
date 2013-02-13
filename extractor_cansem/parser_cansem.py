import argparse
import sys
import os
import subprocess

from annotated_set import loadMRFile
from data_structures import CanonicalDerivation
from canonical_parser import CanonicalParser
from derivation_tree import DerivationTree

from conversion.ghkm2tib import ghkm2tib
#from lib.amr.dag import Dag

class ParserCanSem:

    def __init__(self):
        pass

    @classmethod
    def help(self):
        """
        Returns CanSem help message.
        """
        return ParserCanSem.main(ExtractorCanSem(),"--help")

    def main(self, *args):
        parser = argparse.ArgumentParser(description='CanSem Extraction Algorithm for SHRG',
                                         fromfile_prefix_chars='@',
                                         prog='%s extract-cansem'%sys.argv[0])

        parser.add_argument('grammar', type=str, help="Grammar File")
        parser.add_argument('input', type=str, help="Input File")
        parser.add_argument('--tiburonLoc', nargs='?', default='/home/kmh/Files/Tools/newtib/tiburon', help="Tiburon executable file")
        parser.add_argument('--prefix', nargs='?', default=False, help="Prefix for temporary and output files")

        args = parser.parse_args(args=args)

        if args.prefix == False:
            args.prefix = "test"
        args.output_path = "%s.out"%args.prefix

        # load input data into AnnotatedSet

        data = loadMRFile(args.input)

        derivations = []
        for sentence in data:
            # Extraction
            parser = CanonicalParser(sentence)
            if len(parser.derivs_done) > 0:
                derivations.append((sentence,parser.derivs_done[0]))

        print len(derivations)
        self.parseMRfiles(args,derivations)

    def parseMRfiles(self,args,derivations):

        output_file = open(args.output_path,'w')

        for s,d in derivations:
            x = DerivationTree.fromDerivation(d)
            tibTree = x.getTiburonTree()
            genString = tibTree2String(args,tibTree)
            if genString: 
                output_file.write("%s\n"%genString)
                print genString

        output_file.close()

def tibTree2String(args,tree):
    #cat mnew/tree4.file | ./tiburon - ../stanford-ghkm-2010-03-08/out.tib -k 5

    cmd = [args.tiburonLoc,'-',args.grammar,'-k 1'] #,'-k %d'%args.tib_sentenceK]
    print cmd, tree
    proc = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    output = proc.communicate("%s\n"%tree)[0].split('\n')
    #clean = defaultdict(float)
    if proc.returncode == 0:
        for sentence in output:
            return sentence # simply return the top sentence, ignore all else at this point
            #try:
            #    sentence = tibFormat(sentence,reverse=True)
            #    x,y = sentence.split('#')
            #    clean[x] += float(y)
            #except:
            #    pass
    return False #clean
